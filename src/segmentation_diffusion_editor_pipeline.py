from dataclasses import dataclass
from typing import List, Optional

import torch
from PIL import Image

from diffusers.utils import BaseOutput

from attr_functions import AttrFunc
from diffusion import DiffusionSynthesizer
from segmentation_model import SegmentationModel
from mask_creator import MaskCreator
from stable_diffusion_wrapper import StableDiffusionWrapper


from transforms import tensor_to_pil, tensors_to_pils, pil_to_tensor
from utils import apply_mask, get_device, generate_random_samples


@dataclass
class EditorOutput(BaseOutput):
    imgs: Image.Image
    pred_original_samples: Optional[List[Image.Image]] = None
    model_outputs: Optional[List[torch.Tensor]] = None
    segmentation: Optional[List[Image.Image]] = None
    mask: Optional[List[Image.Image]] = None
    pts: Optional[List[Image.Image]] = None


class SegDiffEditPipeline:
    def __init__(
        self,
        diffusion_wrapper: DiffusionSynthesizer | StableDiffusionWrapper,
        segmentation_model: SegmentationModel,
    ):
        self.diffusion_wrapper = diffusion_wrapper
        self.segmentation_model = segmentation_model
        self.device = get_device()

    # move to helper functions or make it part of the decode call?
    def decode_batch_of_tensors(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        return [self.diffusion_wrapper.decode(t.unsqueeze(0)) for t in tensor]

    # move to helper functions or make it part of the decode call?
    def process_lists_of_tensors_and_decode(
        self, tensors: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        tensor_stacked = torch.stack(tensors, dim=0).squeeze()
        return self.decode_batch_of_tensors(tensor_stacked)

    def edit_image(
        self,
        img: Image.Image,
        xt: torch.Tensor | None,
        model_outputs: List[torch.Tensor] | None,
        eta: float = 0,
        zs: List[torch.Tensor] = [],
        classes: List[int] = [],
        resynthesize: bool = False,
        attr_func: AttrFunc | None = None,
        apply_mask_with_attr_func: bool = False,
        dilate_mask: bool = False,
        prompt: str = "",
        guidance_scale: float = 7.5,
    ) -> EditorOutput:
        # 1. apply mask on cls area to change it (resynthesize or not)
        # 2. apply a strategy, with masking, i.e. to make eyes blue
        # 3. apply a strategy, nothing else, i.e. apply the strategy to the whole image
        # If you want to resynthesize the cls area and make a color change, it is
        # better to first resynthesize the area to obtain a new x_0 and then edit that image with a color change.
        # Alternatively, you can recompute the mask after some timestep t before applying the color change.

        # This method does the following:
        # 1. Segments the image.
        # 2. Creates a mask from the segmentation.
        # 3. If xt is none, it infers it using the reverse inverse process
        # 4. Makes the necessary changes to xt and zs using the mask.
        # 5. Runs the diffusion process on the modified xt and zs and
        # applies an attr function at each timestep.

        if zs and eta == 0 or not zs and eta > 0:
            raise ValueError(
                "If zs is not empty, eta must be greater than 0. If zs is empty, eta must be 0"
            )

        if not resynthesize and attr_func is None:
            raise ValueError(
                "If resynthesize is False, attr_func must be provided. Or no edit is made"
            )

        if not classes and attr_func is None:
            raise ValueError(
                "If classes is empty, attr_func must be provided. Or not edit is made"
            )

        latent = pil_to_tensor(img).to(self.device)
        segmentation = self.segmentation_model(latent)

        if self.diffusion_wrapper.vae is not None:
            latent = self.diffusion_wrapper.encode(latent)

        # infer xt if not provided, mostly for real images, but also works for synthetic images
        if xt is None and eta > 0:
            raise NotImplementedError(
                "If xt is not provided, eta must be 0. Reverse ODE only for deterministic path"
            )
        elif xt is None and eta == 0:
            xt, model_outputs = self.diffusion_wrapper.reverse_inverse_process(
                latent,
                inversion=True,
                prompt=prompt,
                guidance_scale=guidance_scale,
            )
            model_outputs = model_outputs[::-1]

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
            xtv = generate_random_samples(1, self.diffusion_wrapper.unet)

            if isinstance(xt, torch.Tensor):
                xt = apply_mask(mask, xt, xtv)  # type: ignore

            if eta > 0:
                if not zs:
                    raise ValueError("eta > 0 and zs is empty")
                zsv = generate_random_samples(
                    self.diffusion_wrapper.scheduler.num_inference_steps,
                    self.diffusion_wrapper.unet,
                )
                zs = apply_mask(mask, zs, zsv)  # type: ignore

        new_model_outputs = list()
        pred_original_samples = list()
        pts = list()

        additional_setup = self.diffusion_wrapper._additional_setup(
            prompt=prompt, guidance_scale=guidance_scale
        )

        for step_idx, timestep in enumerate(self.diffusion_wrapper.scheduler.timesteps):
            model_output = self.diffusion_wrapper.predict_model_output(
                xt, timestep, **additional_setup
            )
            if mask is not None and model_outputs is not None:
                model_output = apply_mask(mask, model_outputs[step_idx], model_output)

            new_model_outputs.append(model_output)

            variance_noise = self.diffusion_wrapper.get_variance_noise(
                zs, step_idx, eta
            )
            xt, pred_original_sample = self.diffusion_wrapper.single_step(
                model_output, timestep, xt, eta, variance_noise
            )
            if attr_func is not None:
                mask_to_use = (
                    mask if apply_mask_with_attr_func and mask is not None else None
                )
                xt, pt = attr_func.apply(
                    input_image=xt,
                    model_output=model_output,  # type: ignore
                    timestep=timestep,  # type: ignore
                    step_idx=step_idx,
                    scheduler=self.diffusion_wrapper.scheduler,
                    model=self.diffusion_wrapper,
                    mask=mask_to_use,
                    x_0=latent,
                )
            pred_original_samples.append(pred_original_sample)
            pts.append(pt)

        tensor = self.diffusion_wrapper.decode(xt)
        img = tensor_to_pil(tensor)

        pred_original_samples = self.process_lists_of_tensors_and_decode(
            pred_original_samples
        )
        pred_original_samples = tensors_to_pils(pred_original_samples)

        pts = self.process_lists_of_tensors_and_decode(pts)
        pts = tensors_to_pils(pts)

        segmentation = tensor_to_pil(segmentation)
        mask = tensor_to_pil(mask) if mask is not None else None

        return EditorOutput(
            img, pred_original_samples, new_model_outputs, segmentation, mask, pts
        )
