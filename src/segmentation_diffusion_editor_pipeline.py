from dataclasses import dataclass
from typing import List, Optional, Union, Tuple

import torch
from PIL import Image

from diffusers import (
    DDIMScheduler,
    UNet2DModel,
    VQModel,
)
from diffusers.utils import BaseOutput

from attr_functions import AttrFunc
from diffusion_synthesizer import DiffusionSynthesizer
from segmentation_model import SegmentationModel
from mask_creator import MaskCreator

from transforms import (
    get_image_transform,
    image_transform,
    get_reverse_image_transform,
    reverse_transform,
)
from utils import apply_mask, generate_random_samples


# should I take model_outputs as a parameter in case I want to regenerate a particular image?
def reverse_inverse_process(
    x_t: torch.Tensor,
    unet: UNet2DModel,
    scheduler: DDIMScheduler,
    inversion: bool = False,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    timesteps = scheduler.timesteps
    model_outputs = list()

    if inversion:
        timesteps = reversed(timesteps)

    for step_idx, step_time in enumerate(timesteps):
        with torch.no_grad():
            model_output = unet(x_t, step_time).sample
        model_outputs.append(model_output)

        # compute alphas, betas
        alpha_prod_t = scheduler.alphas_cumprod[step_time]
        beta_prod_t = 1 - alpha_prod_t

        # Compute predicted x_0
        p_t = (x_t - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        if scheduler.config.clip_sample:
            p_t = p_t.clamp(
                -scheduler.config.clip_sample_range, scheduler.config.clip_sample_range
            )

        if inversion:
            next_timestep = min(
                scheduler.config.num_train_timesteps - 2,
                step_time + scheduler.config.num_train_timesteps // scheduler.num_,
            )
            alpha_prod_t_next = (
                scheduler.alphas_cumprod[next_timestep]
                if next_timestep >= 0
                else scheduler.final_alpha_cumprod
            )
            pred_sample_direction = (1 - alpha_prod_t_next) ** (0.5) * model_output
            x_t = alpha_prod_t_next ** (0.5) * p_t + pred_sample_direction
        else:
            prev_timestep = (
                step_time
                - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
            )
            alpha_prod_t_prev = (
                scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else scheduler.final_alpha_cumprod
            )
            pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * model_output
            x_t = alpha_prod_t_prev ** (0.5) * p_t + pred_sample_direction

    return x_t, model_outputs


@dataclass
class EditorOutput(BaseOutput):
    imgs: Image.Image
    residuals: List[torch.Tensor]
    segmentation: torch.Tensor | None
    mask: torch.Tensor | None

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())

            if k == "segmentation" or k == "mask":
                reverse_transform_fn = get_reverse_image_transform()

                return reverse_transform(inner_dict[k], reverse_transform_fn)
            else:
                return inner_dict[k]
        else:
            return self.to_tuple()[k]


class SegDiffEditPipe(DiffusionSynthesizer):
    def __init__(
        self,
        unet: UNet2DModel,
        scheduler: DDIMScheduler,
        segmentation_model: SegmentationModel,
        mask_creator: MaskCreator,
        vae: Optional[VQModel] = None,
    ):
        self.unet = unet
        self.scheduler = scheduler
        self.vae = vae
        self.segmentation_model = segmentation_model
        self.mask_creator = mask_creator

    def determine_where_to_edit(self, image, classes) -> Union[None, torch.Tensor]:
        # if a class is specified, it means we should do something with a mask
        # that could be a masking operation or applying a attr_func to x_t with the mask
        use_mask: bool = True if classes else False
        if use_mask:
            mask = self.mask_creator.create_mask(image, classes=classes)
            return mask
        else:
            return None

    def determine_what_to_edit(
        self,
        xt: torch.Tensor,
        zs: List[torch.Tensor],
        eta: float,
        resynthesize: bool,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # if no mask, x_t and zs are not to be changed with a masking operation
        # if resynthesize, then x_t and zso are edited with a masking operation using x_tv, zsv
        if isinstance(mask, torch.Tensor) and resynthesize:
            xtv = generate_random_samples(1, self.unet)

            if isinstance(xt, torch.Tensor):
                xt = apply_mask(mask, xt, xtv)  # type: ignore

            if eta > 0:
                if not zs:
                    raise ValueError("eta > 0 and zso is empty")
                zsv = generate_random_samples(
                    self.scheduler.num_inference_steps,
                    self.unet,
                )
                zs = apply_mask(mask, zs, zsv)  # type: ignore

        return xt, zs

    def edit(
        self,
        img: Image.Image,
        xt: torch.Tensor,
        eta: int = 0,
        model_outputs: List[torch.Tensor] = [],
        zs: List[torch.Tensor] = [],
        classes: List[int] = [],
        resynthesize: bool = False,
        attr_func: Optional[AttrFunc] = None,
        apply_mask_with_attr_func: Optional[bool] = False,
        guidance: Optional[float] = 1.0,
        # steps: Optional[int] = 50,
    ) -> EditorOutput:
        # 1. apply mask on cls area to change it
        # 2. apply a strategy, with masking, i.e. to make eyes blue, without resynthesizing cls area
        # 3. apply a strategy, nothing else, i.e. apply the strategy to the whole image
        # 4. apply a strategy, with masking, and resynthesize cls area with mask (this is does not work well)
        # better to first resynthesize the area to obtain a new x_0 and then edit that image with a color change.
        # Alternatively, you recompute the mask halfway through when pred_x_0 is of reasonable quality to then
        # start applying the strategy to the cls area.

        # This method does the following:
        # determine where to edit -> mask or not
        # determine what to edit -> x_t, zs, eta, resynthesize
        # determine how to edit -> strategy, apply_mask_to_strategy
        # apply edit -> image_synthesizer.synthesize_image

        # self.scheduler.set_timesteps(steps)

        # get image in tensor form
        latent = image_transform(img, get_image_transform()).to("cuda")

        # create segmentation
        segmentation = self.segmentation_model(latent)

        # create mask if necessary
        mask = self.determine_where_to_edit(segmentation, classes)

        # encode latent if working with latent diffusion model
        if self.vae is not None:
            latent = self.encode(latent)

        # infer xt if not provided, mostly for real images, but also works for synthetic images
        if xt is None and eta == 0:
            xt, model_outputs = reverse_inverse_process(
                latent,
                self.unet,
                self.scheduler,
                inversion=True,
            )
            model_outputs = model_outputs[::-1]

        # change xt and zs if eta > 0 (zs must be provided), and only if resynthesize is True
        xt, zs = self.determine_what_to_edit(xt, zs, eta, resynthesize, mask)

        img, model_outputs = self.synthesize_image(
            xt=xt,
            x_0=latent,
            zs=zs,
            eta=eta,
            model_outputs=model_outputs,
            attr_func=attr_func,
            guidance=guidance,
            mask=mask,
            apply_mask_with_attr_func=apply_mask_with_attr_func,
        )

        return EditorOutput(img, model_outputs, segmentation, mask)
