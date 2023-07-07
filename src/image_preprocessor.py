from typing import Tuple
import torch
import torchvision.transforms as transforms

from utils import get_device


class ImagePreprocessor:
    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
    ) -> None:
        self.device = get_device()
        self.to_tensor = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def process(self, image: torch.Tensor) -> torch.Tensor:
        return self.to_tensor(image).to(self.device)
