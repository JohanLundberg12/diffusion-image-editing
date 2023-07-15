from typing import Optional
import torch

from diffusers import (
    DDIMScheduler,
    UNet2DModel,
    VQModel,
)

from base_diffusion import BaseDiffusion


class DiffusionSynthesizer(BaseDiffusion):
    def __init__(
        self, unet: UNet2DModel, scheduler: DDIMScheduler, vae: Optional[VQModel] = None
    ):
        super().__init__(unet, scheduler, vae)

    def encode(self, sample: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            return sample
        sample = sample.to(dtype=torch.float32)
        latent = self.vae.encode(sample).latents  # type: ignore

        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            return latent
        latent = latent.to(dtype=torch.float32)
        sample = self.vae.decode(latent).sample  # type: ignore

        return sample

    def predict_model_output(
        self, xt: torch.Tensor, timestep: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        with torch.no_grad():
            model_output = self.unet(xt, timestep).sample

        return model_output
