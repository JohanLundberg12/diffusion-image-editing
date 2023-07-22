from typing import Optional, Union
import torch

from diffusers import (
    DDIMScheduler,
    UNet2DConditionModel,
    UNet2DModel,
)

from utils import create_progress_bar, get_device, initialize_random_samples, set_seed


class Diffusion:
    def __init__(
        self,
        model,
    ) -> None:
        self.device = get_device()
        self.model = model
        self.unet: Union[UNet2DConditionModel, UNet2DModel] = self.model.unet
        self.scheduler: DDIMScheduler = self.model.scheduler
        self.data_dimensionality = self.unet.sample_size

    def encode(self, sample: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def generate_image(
        self,
        xt,
        eta,
        zs,
        num_inference_steps,
        generator=None,
        **kwargs,
    ):
        raise NotImplementedError

    def generate_images(
        self,
        num_images: int = 1,
        eta: float = 0,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        show_progbar: bool = True,
        return_pred_original_samples: bool = True,
        prompt: str = "",
        cfg_scale: float = 3.5,
    ):
        generator = set_seed(seed)

        self.scheduler.set_timesteps(num_inference_steps)

        all_imgs = []
        all_model_outputs = []
        all_original_sample_preds = []

        pbar = create_progress_bar(range(num_images), show_progbar)

        for i in pbar:
            xt, zs = initialize_random_samples(
                self.model,
                num_inference_steps=num_inference_steps,
                eta=eta,
                generator=generator,
            )
            sample, model_outputs, pred_original_samples = self.generate_image(
                xt=xt,
                eta=eta,
                zs=zs,
                num_inference_steps=num_inference_steps,
                generator=generator,
                prompt=prompt,
                cfg_scale=cfg_scale,
            )
            all_imgs.append(sample)
            all_model_outputs.append(model_outputs)

            if return_pred_original_samples:
                all_original_sample_preds.append(pred_original_samples)

        return all_imgs, all_model_outputs, all_original_sample_preds
