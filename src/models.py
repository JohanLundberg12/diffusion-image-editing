from typing import Any, Dict, Union, Tuple

from diffusers import (
    DDIMScheduler,
    UNet2DModel,
    VQModel,
)

from wrapper import StableDifussionWrapper


def load_unet(
    model_id: str = "google/ddpm-celebahq-256",
    device: str = "cuda",
) -> UNet2DModel:
    unet = UNet2DModel.from_pretrained(model_id)
    unet.to(device)  # type: ignore

    return unet  # type: ignore


def load_vqmodel(
    model_id: str = "CompVis/ldm-celebahq-256",
    device: str = "cuda",
) -> VQModel:
    vq = VQModel.from_pretrained(model_id)
    vq.to(device)  # type: ignore

    return vq  # type: ignore


def get_scheduler(
    model_id: Dict[str, Any] = "google/ddpm-celebahq-256",  # type: ignore
    timesteps: int = 50,
    sample_clipping=True,
) -> DDIMScheduler:
    scheduler = DDIMScheduler.from_pretrained(model_id)
    scheduler.set_timesteps(timesteps)

    if not sample_clipping:
        # real image editing using DDPM won't work if not False, ddpm was trained with this flag=True
        # but it needs/should to be False in real image editing
        # LDM was trained with this flag=False
        scheduler.config.clip_sample = False
    else:
        scheduler.config.clip_sample = True

    return scheduler


def load_modules(
    name: str,
    device: str = "cuda",
    timesteps: int = 50,
    sample_clipping=True,
) -> Tuple[UNet2DModel, DDIMScheduler, Union[VQModel, None]]:
    if name == "ddpm":
        unet = load_unet(model_id="google/ddpm-celebahq-256", device=device)
        scheduler = get_scheduler(
            model_id="google/ddpm-celebahq-256",  # type: ignore
            timesteps=timesteps,
            sample_clipping=sample_clipping,
        )

        return unet, scheduler, None

    elif name == "ldm":
        unet = load_unet(model_id="CompVis/ldm-celebahq-256", device=device)
        scheduler = get_scheduler(
            model_id="CompVis/ldm-celebahq-256",  # type: ignore
            timesteps=timesteps,
            sample_clipping=sample_clipping,
        )
        vqmodel = load_vqmodel(model_id="CompVis/ldm-celebahq-256", device=device)

        return unet, scheduler, vqmodel

    elif name == "sd":
        pass
    else:
        raise ValueError(f"Unknown model name: {name}")
