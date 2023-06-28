import os.path as osp

import torch

# Local application/library specific imports
# ! git clone https://github.com/JohanLundberg12/DDIM-Segmentation.git
from DDIMSegmentation.model import BiSeNet
from diffusers import DDIMPipeline  # type: ignore

from utils import get_device


def load_model(
    model_id: str = "google/ddpm-celebahq-256",
    device: str = "cuda",
    timesteps: int = 50,
    sample_clipping=True,
) -> DDIMPipeline:
    """
    Loads a pretrained model from the HuggingFace model hub.
    """
    if not device:
        device = get_device(verbose=True)
    pipeline = DDIMPipeline.from_pretrained(model_id)
    pipeline.to(device)  # type: ignore
    pipeline.scheduler.set_timesteps(timesteps)  # type: ignore

    if not sample_clipping:
        pipeline.scheduler.config.clip_sample = (
            False  # won't work without this, maybe model was trained with this flag
        )
    else:
        pipeline.scheduler.config.clip_sample = True

    return pipeline


def load_net(
    ckpt="79999_iter.pth",
    n_classes=19,
    device="cuda",
    ckpt_dir="DDIMSegmentation/res/cp",
) -> BiSeNet:
    if not device:
        device = get_device(verbose=True)
    net = BiSeNet(n_classes=n_classes)
    net.to(device)

    # If the directory part is non-empty,
    # it means that ckpt contains a longer path and not just the filename
    if osp.dirname(ckpt):
        save_pth = ckpt  # use as pth
    else:
        save_pth = osp.join(ckpt_dir, ckpt)
    net.load_state_dict(torch.load(save_pth))  # type: ignore
    net.eval()

    return net
