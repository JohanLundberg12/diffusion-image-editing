from typing import List, Optional, Tuple, Union
import numpy as np
import random
import torch

from attr_func_strategy import AttrFuncStrategy
from image_synthesizer import ImageSynthesizer
from mask_handler import MaskHandler

from real_image_editing_utils import (
    reverse_inverse_process,
)

from utils import (
    generate_random_samples,
)


class ImageEditor:
    def __init__(
        self,
        image_synthesizer: ImageSynthesizer,
        mask_handler: MaskHandler,
        seed: Optional[int] = None,
    ):
        self.image_synthesizer = image_synthesizer
        self.mask_handler = mask_handler
        self.image_synthesizer.mask_handler = mask_handler

        if seed:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def get_image_latents(self, x_0: torch.Tensor):
        x_t, residuals = reverse_inverse_process(
            x_0, self.image_synthesizer.model, inversion=True
        )
        return x_t, residuals

    def edit_real_image(
        self,
        x_0: torch.Tensor,
        classes: Optional[List[int]] = [],
        resynthesize: bool = False,
        strategy: Optional[AttrFuncStrategy] = None,
        apply_mask_to_strategy: bool = False,
    ):
        # given a real image, the latent variable x_t is inferred
        # x_0 and the residuals are then obtained from the diffusion model
        # the edit is then applied to the latent variable x_t and residuals
        x_t, residuals = self.get_image_latents(x_0)
        x_0, residuals = self.image_synthesizer.synthesize_image(
            x_t,
            eta=0,
            residuals=residuals,
            strategy=strategy,
        )

        x_edit, residuals = self.edit_image(
            x_0=x_0,
            x_t=x_t,
            eta=0,
            residuals=residuals,
            classes=classes,
            resynthesize=resynthesize,
            strategy=strategy,
            apply_mask_to_strategy=apply_mask_to_strategy,
        )

        return x_edit, residuals

    def edit_image(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        eta: float = 0,
        zs: List[torch.Tensor] = [],
        residuals: List[torch.Tensor] = [],
        classes: Optional[List[int]] = [],
        resynthesize: bool = False,
        strategy: Optional[AttrFuncStrategy] = None,
        apply_mask_to_strategy: bool = False,
    ):
        # 1. apply mask on cls area to change it
        # 2. apply a strategy, with masking, i.e. to make eyes blue, without resynthesizing cls area
        # 3. apply a strategy, nothing else, i.e. apply the strategy to the whole image
        # 4. apply a strategy, with masking, and resynthesize cls area with mask (this is does not work well)
        # better to first resynthesize the area to obtain a new x_0 and then edit that image with a color change.
        # Alternatively, you recompute the mask halfway through when pred_x_0 is of reasonable quality to then
        # start applying the strategy to the cls area.

        # This method does the following:
        # determine where to edit
        # determine what to edit
        # determine how to edit
        # apply edit

        self.determine_where_to_edit(x_0, classes)
        x_t, zs = self.determine_what_to_edit(x_t, zs, eta, resynthesize)
        strategy = self.determine_how_to_edit(strategy, apply_mask_to_strategy)

        x_edit, residuals = self.image_synthesizer.synthesize_image(
            x_t=x_t,
            zs=zs,
            eta=eta,
            residuals=residuals,
            strategy=strategy,
        )

        return (
            x_edit,
            residuals,
            self.mask_handler.get_segmentation(),  # type: ignore
            self.mask_handler.get_mask(),  # type: ignore
        )

    def determine_where_to_edit(self, image, classes) -> None:
        # if a class is specified, it means we should do something with a mask
        # that could be a masking operation or applying a strategy to x_t with the mask
        use_mask: bool = True if classes else False
        if use_mask:
            if self.mask_handler.mask is None:
                self.mask_handler.create_mask(image, classes=classes)

    def determine_what_to_edit(
        self, x_t, zs, eta, resynthesize
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # if no mask, x_t and zs are not to be changed with a masking operation
        # if resynthesize, then x_t and zso are changed
        if isinstance(self.mask_handler.mask, torch.Tensor):
            if resynthesize:
                x_tv = generate_random_samples(1, self.image_synthesizer.model)
                x_t = self.mask_handler.apply_mask(self.mask_handler.mask, x_t, x_tv)

                if eta > 0:
                    if not zs:
                        raise ValueError("eta > 0 and zs is empty")
                    zsv = generate_random_samples(
                        self.image_synthesizer.model.scheduler.num_inference_steps,
                        self.image_synthesizer.model,
                    )
                    zs = [
                        self.mask_handler.apply_mask(self.mask_handler.mask, zo, zv)
                        for zo, zv in zip(zs, zsv)
                    ]
        return x_t, zs

    def determine_how_to_edit(
        self, strategy: AttrFuncStrategy, apply_method_to_strategy: bool
    ) -> Union[AttrFuncStrategy, None]:
        # if a strategy is specified and a mask is created and the strategy
        # is intended to be used with a mask, then we set the mask on the strategy
        if strategy and self.mask_handler.mask is not None and apply_method_to_strategy:
            strategy.mask = self.mask_handler.mask
        return strategy
