from typing import List, Union
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torchvision.transforms import Lambda


def _scale_zero_one(input: torch.Tensor) -> torch.Tensor:
    return (input / 2 + 0.5).clamp(0, 1)


def tensor_to_pil(tensor_img: torch.Tensor) -> Image.Image:
    transform_normal = transforms.Compose(
        [
            transforms.Lambda(_scale_zero_one),
            Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            Lambda(lambda t: t * 255.0),  # back to 0-255
            Lambda(lambda t: t.cpu().detach().numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ]
    )
    transform_mask = transforms.Compose(
        [
            Lambda(lambda t: t.cpu().to(torch.uint8)),
            transforms.ToPILImage(),
        ]
    )

    if tensor_img.dim() == 2:
        pil_img = transform_mask(tensor_img)
    elif tensor_img.dim() == 3:
        pil_img = transform_normal(tensor_img)
    elif tensor_img.dim() == 4:
        assert tensor_img.shape[0] == 1
        pil_img = transform_normal(tensor_img.squeeze(0))
    return pil_img


def tensors_to_pils(tensor_imgs: List[torch.Tensor]) -> List[Image.Image]:
    pil_imgs = list()

    for img in tensor_imgs:
        pil_image = tensor_to_pil(img)
        pil_imgs.append(pil_image)

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
