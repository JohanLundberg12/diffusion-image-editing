from dataclasses import dataclass
from typing import List, Optional

import torch
from PIL import Image

from diffusers.utils import BaseOutput

from attr_functions import AttrFunc
from diffusion_classes import DDPM, LDM, SD
from mask_creator import MaskCreator
from models import SegmentationModel

from ddpm_inversion import invert, reverse_step
from ddim_inversion import ddim_inversion
from diffusion_utils import (
    diffusion_loop,
    get_noise_pred,
    get_variance_noise,
    single_step,
)
from transforms import tensor_to_pil
from utils import (
    apply_mask,
    get_device,
    generate_random_samples,
    process_lists_of_tensors,
)

from constants import ATTRS


@dataclass
class EditorOutput(BaseOutput):
    imgs: Image.Image
    pred_original_samples: Optional[List[Image.Image]] = None
    model_outputs: Optional[List[torch.Tensor]] = None


class SegDiffEditPipeline:
    """
    SegDiffEditPipeline edits an image using a segmentation model and a diffusion model.
    The segmentation model creates a mask from the image which can be combined with custom
    attribute funcions that guide the image editing process. Real images are inverted using
    the ddim or ddpm inversion methods. Noise maps xt, zs and model outputs can be edited
    by applying the mask on them in combination with other random noise maps. The diffusion
    process is run on the modified noise maps and the attribute function is applied at each
    timestep if specified.
    """

    def __init__(
        self,
        diffusion_wrapper: DDPM | LDM | SD,
        segmentation_model: SegmentationModel,
    ):
        self.diffusion_wrapper = diffusion_wrapper
        self.segmentation_model = segmentation_model

        self.device = get_device()

    def check_classes(self, classes):
        for x in classes:
            assert 0 <= x < len(ATTRS)

    def check_inputs(self, attr_func, eta, mask, resynthesize, zs):
        if eta > 0 and zs is None:
            raise ValueError("eta > 0 and zs is empty")

        if zs is not None and eta == 0:
            raise ValueError("eta == 0 and zs is not empty")

        if attr_func is None:
            if mask is None or resynthesize is None:
                raise ValueError(
                    "attr_func is None and classes and mask is None implies no edit"
                )

    # call this method before edit_image
    def prepare_for_edit(
        self,
        img: torch.Tensor,
        classes: Optional[List[int]] = None,
        dilate_mask: bool = False,
    ):
        self.check_classes(classes)

        if classes is not None:
            segmentation = self.segmentation_model(img)
            mask = self.create_mask(classes, dilate_mask, segmentation)
        else:
            segmentation = None
            mask = None

        latent = self.diffusion_wrapper.encode(img)

        return latent, mask, segmentation

    def edit_noise_map(
        self,
        noise_map: torch.Tensor,
        mask: torch.Tensor,
    ):
        num_samples = noise_map.shape[0]
        noise_map_random = generate_random_samples(
            num_samples, self.diffusion_wrapper.model.unet
        ).to(self.device)

        noise_map = apply_mask(mask, noise_map, noise_map_random)

        return noise_map

    def edit_noise_maps(self, xt, zs, mask, resynthesize):
        if mask is not None and resynthesize is not None:
            xt = self.edit_noise_map(xt, mask)

            if zs is not None:
                zs = self.edit_noise_map(zs, mask)

        return xt, zs

    def create_mask(self, classes, dilate_mask, segmentation):
        mask_creator = MaskCreator(
            dilate_mask=dilate_mask,
            resize_size=(
                self.diffusion_wrapper.data_dimensionality,
                self.diffusion_wrapper.data_dimensionality,
            ),
        )
        mask = mask_creator.create_mask(segmentation, classes=classes)

        return mask

    def prepare_text_emb(self, prompt):
        if prompt is None:
            return None
        else:
            return self.diffusion_wrapper.additional_prep(
                self.diffusion_wrapper.model, prompt
            )

    def postprocess(self, xt, pred_original_samples):
        tensor = self.diffusion_wrapper.decode(xt)
        img = tensor_to_pil(tensor)

        pred_original_samples = process_lists_of_tensors(
            self.diffusion_wrapper, pred_original_samples
        )

        return img, pred_original_samples

    def prepare_real_image_edit(
        self,
        img: torch.Tensor,
        eta: float = 0,
        inversion_method: str = "ddim",
        classes: Optional[List[int]] = None,
        dilate_mask: bool = False,
        prompt: Optional[str] = None,
        cfg_scale: Optional[float] = None,
        prog_bar: bool = True,
    ):
        if inversion_method == "ddim" and eta > 0:
            raise ValueError("eta > 0 and inversion_method == 'ddim' is not possible")

        latent, mask, segmentation = self.prepare_for_edit(img, classes, dilate_mask)

        if type(self.diffusion_wrapper) == DDPM:
            assert self.diffusion_wrapper.model.scheduler.config.clip_sample is False
        elif type(self.diffusion_wrapper) == LDM:
            assert self.diffusion_wrapper.model.scheduler.config.clip_sample is False

        # inversion x_t <- x_0
        if inversion_method == "ddim":
            xt = ddim_inversion(
                model=self.diffusion_wrapper.model,
                x0=latent,
                prompt=prompt,
                cfg_scale=cfg_scale,
            )
            xts = None
            zs = None
        elif inversion_method == "ddpm":
            xt, zs, xts = invert(
                model=self.diffusion_wrapper.model,
                x0=latent,
                num_inference_steps=self.diffusion_wrapper.scheduler.num_inference_steps,
                eta=eta,
                prompt=prompt,
                cfg_scale=cfg_scale,
                prog_bar=prog_bar,
            )
        else:
            raise ValueError(f"Unknown inversion method: {inversion_method}")

        return xt, zs, xts, mask, segmentation

    def edit_image(
        self,
        xt: torch.Tensor,
        eta: float = 0,
        model_outputs: Optional[List[torch.Tensor]] = None,
        zs: Optional[torch.Tensor] = None,
        xts: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        attr_func: AttrFunc | None = None,
        prompt: Optional[str] = None,
        cfg_scale: Optional[float] = None,
        inversion_method: str = "ddim",
        Tskip: Optional[int] = None,
        resynthesize: bool = False,
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

        # Check inputs are valid
        self.check_inputs(
            attr_func=attr_func, eta=eta, mask=mask, resynthesize=resynthesize, zs=zs
        )
        xt, zs = self.edit_noise_maps(xt, zs, mask, resynthesize)

        text_emb = self.prepare_text_emb(prompt)

        new_xts = list()
        new_model_outputs = list()
        pred_original_samples = list()

        if xts is not None:
            xt = xts[Tskip].unsqueeze(0)
            zs = zs[Tskip:]

        for step_idx, timestep in diffusion_loop(self.diffusion_wrapper.model, zs):
            # for step_idx, timestep in enumerate(self.diffusion_wrapper.scheduler.timesteps):
            with torch.no_grad():
                noise_pred = get_noise_pred(
                    self.diffusion_wrapper.model, xt, timestep, text_emb, cfg_scale
                )

            if mask is not None:
                noise_pred = apply_mask(mask, model_outputs[step_idx], noise_pred)

            variance_noise = get_variance_noise(zs, step_idx, eta)

            if inversion_method == "ddpm" and Tskip is not None:
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

        img, pred_original_samples = self.postprocess(xt, pred_original_samples)

        return EditorOutput(img, pred_original_samples, new_model_outputs)
