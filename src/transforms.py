from typing import Callable
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torchvision.transforms import Lambda


def input_t(input: torch.Tensor) -> torch.Tensor:
    # Scale between [-1, 1]
    return (2 * input - 1).clamp(-1, 1)  # TODO: clamp (-1, 1) ?


def get_image_transform(image_size: int) -> Callable[[Image.Image], torch.Tensor]:
    """Returns a transform that scales the image pixel values to [-1, 1]"""

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # Shape: HWC, Scales data into [0,1] by div / 255, so does normalization
            transforms.Lambda(input_t),
        ]
    )

    return transform


def output_t(input: torch.Tensor) -> torch.Tensor:
    # Scale between [0, 1]
    return ((input + 1) / 2).clamp(
        0, 1
    )  # TODO: .clamp(0, 1)? and what about input / 2 + 0.5?


def get_reverse_image_transform() -> Callable[[torch.Tensor], Image.Image]:
    """Returns a transform that scales the image pixel values to [0, 255]
    and converts it to a PIL image."""

    reverse_transform = transforms.Compose(
        [
            Lambda(output_t),
            Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            Lambda(lambda t: t * 255.0),  # back to 0-255
            Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ]
    )
    return reverse_transform


def image_transform(
    image: Image.Image, transform: Callable[[Image.Image], torch.Tensor]
) -> torch.Tensor:
    """Transforms an image to a tensor and scales its pixel values to be between -1 and 1."""

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Reshape to (B, C, H, W)

    return image_tensor


def reverse_transform(
    image_tensor: torch.Tensor, transform: Callable[[torch.Tensor], Image.Image]
) -> Image.Image:
    """Transforms a tensor to an image and scales its pixel values to be between 0 and 255."""

    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor.squeeze(0)  # Reshape to (C, H, W)
    elif len(image_tensor.shape) == 3:
        pass  # expected to have dimensions (C, H, W)
    elif len(image_tensor.shape) == 2:
        return transforms.ToPILImage()(image_tensor.cpu().to(torch.uint8))
    else:
        raise ValueError(
            f"Expected image tensor to have 2, 3 or 4 dimensions, but got {len(image_tensor.shape)}"
        )

    image = transform(image_tensor)

    return image
