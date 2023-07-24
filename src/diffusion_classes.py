import torch

from diffusers import (
    AutoencoderKL,
    VQModel,
)

from base_diffusion import Diffusion

from transformers import CLIPTextModel, CLIPTokenizer

from diffusion_utils import prep_text


class SD(Diffusion):
    def __init__(
        self,
        model,
    ):
        super().__init__(model)
        self.vae: AutoencoderKL = self.model.vae
        self.tokenizer: CLIPTokenizer = model.tokenizer
        self.text_encoder: CLIPTextModel = model.text_encoder

    def encode(self, sample: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latent = self.vae.encode(sample).latent_dist.mode().detach()
        return 0.18215 * latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        latent = 1 / 0.18215 * latent

        with torch.no_grad():
            sample = self.vae.decode(latent).sample

        return sample

    def additional_prep(self, model, prompt):
        return prep_text(model, prompt)


class LDM(Diffusion):
    def __init__(
        self,
        model,
    ):
        super().__init__(model)
        self.vae: VQModel = self.model.vae

    def encode(self, sample: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            sample = sample.to(dtype=torch.float32)
            latent = self.vae.encode(sample).latents  # type: ignore

        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latent = latent.to(dtype=torch.float32)
            sample = self.vae.decode(latent).sample  # type: ignore

        return sample


class DDPM(Diffusion):
    def __init__(self, model):
        super().__init__(model)

    def encode(self, sample: torch.Tensor) -> torch.Tensor:
        return sample

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent
