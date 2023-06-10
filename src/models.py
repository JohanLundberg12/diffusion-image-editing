import os.path as osp

import torch

from DDIMSegmentation.model import BiSeNet
from diffusers import DDIMPipeline


def load_model(model_id="google/ddpm-celebahq-256", device="cuda", timesteps=50):
    model = DDIMPipeline.from_pretrained(model_id).to(device)
    model.scheduler.set_timesteps(timesteps)

    return model


def load_net(
    ckpt="79999_iter.pth",
    n_classes=19,
    device="cuda",
    ckpt_dir="DDIMSegmentation/res/cp",
):
    net = BiSeNet(n_classes=n_classes)
    net.to(device)

    # If the directory part is non-empty,
    # it means that ckpt contains a longer path and not just the filename
    if osp.dirname(ckpt):
        save_pth = ckpt  # use as pth
    else:
        save_pth = osp.join(ckpt_dir, ckpt)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    return net
