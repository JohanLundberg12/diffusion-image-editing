import os
from diffusers import (
    AutoencoderKL,
    DDIMPipeline,
    DDIMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)

from transformers import CLIPTextModel, CLIPTokenizer

from diffusion import DiffusionSynthesizer
from stable_diffusion_wrapper import StableDiffusionWrapper

from utils import get_device


class DiffusionModelFactory:
    def __init__(self, sample_clipping=False):
        self.device = get_device()
        self.sample_clipping = sample_clipping

    def create_model(self, name: str) -> DiffusionSynthesizer | StableDiffusionWrapper:
        if name == "ddpm":
            pipe = DDIMPipeline.from_pretrained("google/ddpm-celebahq-256")
            pipe.to(self.device)  # type: ignore

            # editing real images -> set this to False as
            # the reverse inverse process needs this to invert
            # and reconstruct the image properly
            # editing synthetic images, set this to True?
            # DDPM was trained with this flag=True
            pipe.scheduler.config.clip_sample = self.sample_clipping

            return DiffusionSynthesizer(pipe.unet, pipe.scheduler)

        elif name == "ldm":
            pipe = DiffusionPipeline.from_pretrained("CompVis/ldm-celebahq-256")
            pipe.scheduler = DDIMScheduler.from_config(
                "CompVis/ldm-celebahq-256", subfolder="scheduler"
            )
            pipe.to(self.device)  # type: ignore

            # LDM was trained with this flag=False
            pipe.scheduler.config.clip_sample = self.sample_clipping

            return DiffusionSynthesizer(pipe.unet, pipe.scheduler, pipe.vqvae)

        elif name == "sd":
            access_token = os.environ.get("HF_TOKEN")

            vae = AutoencoderKL.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                subfolder="vae",
                use_auth_token=access_token,
            ).to(self.device)

            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            text_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14"
            ).to(self.device)
            token_length = tokenizer.model_max_length

            unet = UNet2DConditionModel.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                subfolder="unet",
                use_auth_token=access_token,
            ).to(self.device)

            scheduler = DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
            )
            return StableDiffusionWrapper(
                unet,
                scheduler,
                vae,
                tokenizer,
                text_encoder,
                token_length,
            )
        else:
            raise ValueError(f"Unknown model name: {name}")
