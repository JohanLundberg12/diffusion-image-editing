from typing import List, Union
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torchvision.transforms import Lambda


def _scale_zero_one(input: torch.Tensor) -> torch.Tensor:
    return (input / 2 + 0.5).clamp(0, 1)


def tensor_to_pil(tensor_imgs):
    transform_normal = transforms.Compose(
        [
            transforms.Lambda(_scale_zero_one),
            Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            Lambda(lambda t: t * 255.0),  # back to 0-255
            Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ]
    )
    transform_mask = transforms.Compose(
        [
            Lambda(lambda t: t.cpu().to(torch.uint8)),
            transforms.ToPILImage(),
        ]
    )

    pil_imgs = list()

    for img in tensor_imgs:
        if len(img.shape) == 4:
            img = img.squeeze(0)  # Reshape to (C, H, W)
            img = transform_normal(img)
        elif len(img.shape) == 3:
            img = transform_normal(img)
        elif len(img.shape) == 2:
            img = transform_mask(img)

        pil_imgs.append(img)

    return pil_imgs


def _scale_minus_one_one(input: torch.Tensor) -> torch.Tensor:
    return (input * 2 - 1).clamp(-1, 1)


def pil_to_tensor(
    pil_imgs: Union[Image.Image, List[Image.Image]]
) -> Union[torch.Tensor, List[torch.Tensor]]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Shape: HWC, Scales data into [0,1] by div / 255, so does normalization
            transforms.Lambda(_scale_minus_one_one),
        ]
    )
    if isinstance(pil_imgs, Image.Image):
        tensor_imgs = transform(pil_imgs)
        tensor_imgs = tensor_imgs.unsqueeze(0)  # Reshape to (B, C, H, W) # type: ignore
    elif isinstance(pil_imgs, list):
        tensor_imgs = torch.cat(
            [transform(img).unsqueeze(0) for img in pil_imgs]  # type: ignore
        )  # Shape: BCHW
    else:
        raise Exception("Input need to be PIL.Image or list of PIL.Image")
    return tensor_imgs
