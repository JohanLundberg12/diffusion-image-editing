import os
import torch
from torchvision import models
from torchvision import transforms
from diffusers import (
    DDIMPipeline,
    DDIMScheduler,
    DiffusionPipeline,
    StableDiffusionPipeline,
)

from Segmentation.model import BiSeNet

from diffusion_classes import DDPM, LDM, SD

from utils import get_device


def create_diffusion_model(name: str, sample_clipping: bool = True) -> DDPM | LDM | SD:
    device = get_device()

    if name == "ddpm":
        model_id = "google/ddpm-celebahq-256"
        model = DiffusionPipeline.from_pretrained(model_id)
        model.scheduler = DDIMScheduler.from_config(model.scheduler.config)
        model.to(device)

        # sample_clip=True for synthetic data, false for real data
        # DDPM was trained with this flag=True
        model.scheduler.config.clip_sample = sample_clipping

        # rescale_betas_zero_snr=True: https://huggingface.co/docs/diffusers/api/schedulers/ddim
        # The paper Common Diffusion Noise Schedules and Sample Steps are Flawed claims that a mismatch between the training and inference settings leads to suboptimal inference generation results for Stable Diffusion.
        # The abstract reads as follows:
        # *We discover that common diffusion noise schedules do not enforce the last timestep to have zero signal-to-noise ratio (SNR)

        return DDPM(model)

    elif name == "ldm":
        model_id = "CompVis/ldm-celebahq-256"
        model = DiffusionPipeline.from_pretrained(model_id).to(device)
        model.scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")

        # LDM was trained with this flag=False
        model.scheduler.config.clip_sample = sample_clipping

        return LDM(model)

    elif name == "sd":
        access_token = os.environ.get("HF_TOKEN")

        model_id = "CompVis/stable-diffusion-v1-4"
        model = StableDiffusionPipeline.from_pretrained(model_id).to(device)
        model.scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")

        return SD(model)
    else:
        raise ValueError(f"Unknown model name: {name}")


def _load_model(model: torch.nn.Module, state_dict_path: str) -> torch.nn.Module:
    """Load a model from a state dict."""

    # load model weights
    state_dict = torch.load(state_dict_path, map_location="cuda")["state_dict"]

    # load weights to model
    model.load_state_dict(state_dict)

    return model


def get_pretrained_anyGAN():
    # manual download of pretrained models:
    # URL = "https://hanlab.mit.edu/projects/anycost-gan/files/attribute_predictor.pt"

    predictor = models.resnet50()
    predictor.fc = torch.nn.Linear(predictor.fc.in_features, 40 * 2)
    predictor = _load_model(predictor, "../attribute_predictor.pt")

    return predictor.to("cuda")


class SegmentationModel:
    def __init__(
        self,
        ckpt: str = "Segmentation/res/cp/79999_iter.pth",
        n_classes: int = 19,
        image_size: tuple = (512, 512),
    ) -> None:
        self.device = get_device()
        self.net = self.load_net(ckpt, n_classes)

        self.to_tensor = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def process(self, image: torch.Tensor) -> torch.Tensor:
        return self.to_tensor(image).to(self.device)

    def load_net(
        self,
        ckpt: str,
        n_classes: int,
    ) -> BiSeNet:
        net = BiSeNet(n_classes=n_classes)
        net.to(self.device)

        net.load_state_dict(torch.load(ckpt))
        net.eval()

        return net

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        t_image = self.process(image)
        out = self.net(t_image)[0]
        segmentation = out.squeeze(0).argmax(0)

        return segmentation
