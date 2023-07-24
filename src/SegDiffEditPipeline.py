from dataclasses import dataclass
from typing import List, Optional

import torch
from PIL import Image

from diffusers.utils import BaseOutput

from attr_functions import AttrFunc
from models import SegmentationModel
from mask_creator import MaskCreator

from diffusion_classes import DDPM, LDM, SD


from ddpm_inversion import invert, reverse_step
from ddim_inversion import ddim_inversion
from diffusion_utils import (
    diffusion_loop,
    get_noise_pred,
    get_variance_noise,
    single_step,
)
from transforms import tensor_to_pil, pil_to_tensor
from utils import apply_mask, get_device, generate_random_samples

from constants import ATTRS


@dataclass
class EditorOutput(BaseOutput):
    imgs: Image.Image
    pred_original_samples: Optional[List[Image.Image]] = None
    model_outputs: Optional[List[torch.Tensor]] = None
    segmentation: Optional[List[Image.Image]] = None
    mask: Optional[List[Image.Image]] = None


class SegDiffEditPipeline:
    def __init__(
        self,
        diffusion_wrapper: DDPM | LDM | SD,
        segmentation_model: SegmentationModel,
    ):
        self.diffusion_wrapper = diffusion_wrapper
        self.segmentation_model = segmentation_model

        self.device = get_device()

    # move to helper functions or make it part of the decode call?
    def to_pil_and_decode_batch_of_tensors(
        self, tensor: torch.Tensor
    ) -> List[Image.Image]:
        return [
            tensor_to_pil(self.diffusion_wrapper.decode(t.unsqueeze(0))) for t in tensor
        ]

    # move to helper functions or make it part of the decode call?
    def process_lists_of_tensors(
        self, tensors: List[torch.Tensor]
    ) -> List[Image.Image]:
        tensor_stacked = torch.stack(tensors, dim=0).squeeze()
        return self.to_pil_and_decode_batch_of_tensors(tensor_stacked)

    def check_inputs(self, attr_func, classes, eta, resynthesize, zs):
        pass

    def edit_image(
        self,
        img: Image.Image,
        xt: torch.Tensor | None,
        model_outputs: List[torch.Tensor] = [],
        eta: float = 0,
        zs: Optional[torch.Tensor] = None,
        classes: List[int] = [],
        resynthesize: bool = False,
        attr_func: AttrFunc | None = None,
        dilate_mask: bool = False,
        prompt: str = "",
        cfg_scale: float = 7.5,
        inversion_method: str = "ddim",
        denoising_method: str = "ddim",
        Tskip: Optional[int] = None,
    ) -> EditorOutput:
        # 1. apply mask on cls area to change it (resynthesize or not)
        # 2. apply an attr_func, with masking, i.e. to make eyes blue
        # 3. apply an attr_func, nothing else, i.e. apply the attr_func to the whole image
        # If you want to resynthesize the cls area and make a color change, it is
        # better to first resynthesize the area to obtain a new x_0 and then edit that image with a color change.
        # Alternatively, you can recompute the mask after some timestep t before applying the color change.

        # This method does the following:
        # 1. Segments the image.
        # 2. Creates a mask from the segmentation.
        # 3. If xt is none, an inversion method inverts the image to noise xt.
        # 4. Makes the necessary changes to xt and zs using the mask.
        # 5. Runs the diffusion process on the modified xt and zs and
        # applies an attr function at each timestep if specified.

        for x in classes:
            assert 0 <= x < len(ATTRS)

        # Check inputs are valid
        self.check_inputs(attr_func, classes, eta, resynthesize, zs)

        # prepare inputs
        tensor = pil_to_tensor(img).to(self.device)
        segmentation = self.segmentation_model(tensor)
        latent = self.diffusion_wrapper.encode(tensor)

        # inversion x_t <- x_0
        if inversion_method == "ddim" and xt is None:
            assert eta == 0

            xt = ddim_inversion(
                model=self.diffusion_wrapper.model,
                x0=latent,
                prompt=prompt,
                cfg_scale=cfg_scale,
            )
            xts = None
        elif inversion_method == "ddpm" and xt is None:
            xt, zs, xts = invert(
                model=self.diffusion_wrapper.model,
                x0=latent,
                num_inference_steps=self.diffusion_wrapper.scheduler.num_inference_steps,
                eta=eta,
                prompt=prompt,
                cfg_scale=cfg_scale,
                prog_bar=True,
            )
        else:
            xts = None

        # only create mask in the case of resynthesize or ColorAttrFunc or NetAttrFunc (not if AnyGANAttrFunc)
        use_mask: bool = True if classes else False
        if use_mask:
            mask_creator = MaskCreator(
                dilate_mask=dilate_mask,
                resize_size=(
                    self.diffusion_wrapper.data_dimensionality,
                    self.diffusion_wrapper.data_dimensionality,
                ),
            )
            mask = mask_creator.create_mask(segmentation, classes=classes)
        else:
            mask = None

        # if no mask, x_t and zs are not to be changed with a masking operation but with a attr_func
        # if resynthesize, then x_t and zso are edited with a masking operation using x_tv, zsv
        if isinstance(mask, torch.Tensor) and resynthesize:
            xtv = generate_random_samples(1, self.diffusion_wrapper.model.unet).to(
                "cuda"
            )

            if isinstance(xt, torch.Tensor):
                xt = apply_mask(mask, xt, xtv)  # type: ignore

            if eta > 0:
                if zs is None:
                    raise ValueError("eta > 0 and zs is empty")
                zsv = generate_random_samples(
                    self.diffusion_wrapper.scheduler.num_inference_steps,
                    self.diffusion_wrapper.model.unet,
                )
                zs = apply_mask(mask, zs, zsv)  # type: ignore

        new_xts = list()
        new_model_outputs = list()
        pred_original_samples = list()

        if prompt is not None:
            text_emb = self.diffusion_wrapper.additional_prep(
                self.diffusion_wrapper.model, prompt
            )
        else:
            text_emb = None

        if xts is not None:
            xt = xts[Tskip]
            zs = zs[Tskip:]

        # for step_idx, timestep in diffusion_loop(self.diffusion_wrapper.model, zs):
        for step_idx, timestep in enumerate(self.diffusion_wrapper.scheduler.timesteps):
            with torch.no_grad():
                noise_pred = get_noise_pred(
                    self.diffusion_wrapper.model, xt, timestep, text_emb, cfg_scale
                )

            if mask is not None:
                noise_pred = apply_mask(mask, model_outputs[step_idx], noise_pred)

            variance_noise = get_variance_noise(zs, step_idx, eta)

            if denoising_method == "ddpm" and Tskip is not None:
                xt = reverse_step(
                    model=self.diffusion_wrapper.model,
                    model_output=noise_pred,
                    timestep=timestep,
                    sample=xt,
                    eta=eta,
                    variance_noise=variance_noise,
                )
            else:  # denoising_method == "ddim"
                xt, pred_original_sample = single_step(
                    model=self.diffusion_wrapper.model,
                    model_output=noise_pred,
                    timestep=timestep,
                    sample=xt,
                    eta=eta,
                    variance_noise=variance_noise,
                )

            # nudge xt if attr_func
            if attr_func is not None:
                if attr_func.kwargs["use_mask"] and mask is not None:
                    attr_func.kwargs["mask"] = mask
                else:
                    attr_func.kwargs["mask"] = None

                xt = attr_func.apply(
                    xt=xt,
                    model_output=noise_pred,
                    timestep=timestep,
                    step_idx=step_idx,
                    model=self.diffusion_wrapper,
                    **attr_func.kwargs,
                )

            new_xts.append(xt)
            new_model_outputs.append(noise_pred)
            pred_original_samples.append(pred_original_sample.detach())

        x0 = xt
        tensor = self.diffusion_wrapper.decode(x0)
        img = tensor_to_pil(tensor)

        pred_original_samples = self.process_lists_of_tensors(pred_original_samples)

        segmentation = tensor_to_pil(segmentation)
        mask = tensor_to_pil(mask) if mask is not None else None

        return EditorOutput(
            img, pred_original_samples, new_model_outputs, segmentation, mask
        )
