import torch

from diffusers import (
    AutoencoderKL,
    VQModel,
)

from base_diffusion import Diffusion

from transforms import tensors_to_pils, tensor_to_pil
from utils import generate_random_samples
from diffusion_utils import diffusion_loop, prep_text

from transformers import CLIPTextModel, CLIPTokenizer


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

    def generate_image(
        self,
        xt,
        eta=0,
        zs=None,
        num_inference_steps=50,
        generator=None,
        prompt="",
        cfg_scale=3.5,
    ):
        self.model.scheduler.set_timesteps(num_inference_steps)

        text_emb = prep_text(self.model, prompt)

        if eta > 0 and zs is None:
            zs = generate_random_samples(
                num_inference_steps, self.model.unet, generator=generator
            )
        x0, model_outputs, pred_original_samples = diffusion_loop(
            self.model,
            xt,
            eta,
            zs,
            text_emb=text_emb,
            cfg_scale=cfg_scale,
        )

        sample = self.decode(x0)

        pred_original_samples = torch.stack(
            pred_original_samples, dim=0
        ).squeeze()  # B, 1, C, H, W -> B, C, H, W

        pred_original_samples = [
            tensor_to_pil(  # to pil to save memory for ldm and sd models
                self.decode(pred_original_sample.unsqueeze(0))  # C, H, W -> 1, C, H, W
            )
            for pred_original_sample in pred_original_samples
        ]

        return sample, model_outputs, pred_original_samples


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

    def generate_image(
        self, xt, eta=0, zs=None, num_inference_steps=50, generator=None
    ):
        self.model.scheduler.set_timesteps(num_inference_steps)

        if eta > 0 and zs is None:
            zs = generate_random_samples(
                num_inference_steps, self.model.unet, generator=generator
            )
        x0, model_outputs, pred_original_samples = diffusion_loop(
            self.model,
            xt,
            eta,
            zs,
        )

        sample = self.decode(x0)

        pred_original_samples = torch.stack(
            pred_original_samples, dim=0
        ).squeeze()  # B, 1, C, H, W -> B, C, H, W

        pred_original_samples = [
            tensor_to_pil(  # to pil to save memory for ldm and sd models
                self.decode(pred_original_sample.unsqueeze(0))  # C, H, W -> 1, C, H, W
            )
            for pred_original_sample in pred_original_samples
        ]

        return sample, model_outputs, pred_original_samples


class DDPM(Diffusion):
    def __init__(self, model):
        super().__init__(model)

    def encode(self, sample: torch.Tensor) -> torch.Tensor:
        return sample

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent

    def generate_image(
        self, xt, eta=0, zs=None, num_inference_steps=50, generator=None, **kwargs
    ):
        self.model.scheduler.set_timesteps(num_inference_steps)

        if eta > 0 and zs is None:
            zs = generate_random_samples(
                num_inference_steps, self.model.unet, generator=generator
            )
        x0, model_outputs, pred_original_samples = diffusion_loop(
            self.model, xt, eta, zs
        )

        img = tensor_to_pil(x0)
        pred_original_samples = torch.stack(
            pred_original_samples, dim=0
        ).squeeze()  # B, 1, C, H, W -> B, C, H, W
        pred_original_samples = tensors_to_pils(pred_original_samples)

        return img, model_outputs, pred_original_samples
