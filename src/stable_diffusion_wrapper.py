import torch

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
)

from transformers import CLIPTextModel, CLIPTokenizer

from base_diffusion import BaseDiffusion


class StableDiffusionWrapper(BaseDiffusion):
    def __init__(
        self,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        vae: AutoencoderKL,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        token_length,
    ):
        super().__init__(unet, scheduler, vae)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.token_length = token_length

    def _additional_setup(self, prompt: str = "", guidance_scale: float = 7.5) -> dict:
        # Get Text Embedding
        text_embeddings = self.prep_text(prompt)
        return {"text_embeddings": text_embeddings, "guidance_scale": guidance_scale}

    def encode(self, sample: torch.Tensor) -> torch.Tensor:
        latent = self.vae.encode(sample).latent_dist.mode().detach()
        return 0.18215 * latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        latent = 1 / 0.18215 * latent
        sample = self.vae.decode(latent).sample

        return sample

    def predict_model_output(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        text_embeddings: torch.Tensor,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> torch.Tensor:
        latent_model_input = torch.cat([latents] * 2)

        with torch.no_grad():
            model_output = self.unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=text_embeddings,
            )["sample"]

        model_output_uncond, model_output_text = model_output.chunk(2)
        model_output = model_output_uncond + guidance_scale * (
            model_output_text - model_output_uncond
        )

        return model_output

    def encode_text(self, prompt: str) -> torch.Tensor:
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.token_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_encoding = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_encoding

    def prep_text(self, prompt: str) -> torch.Tensor:
        # add unconditional embedding
        return torch.cat([self.encode_text(""), self.encode_text(prompt)])
