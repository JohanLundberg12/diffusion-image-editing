from typing import Tuple
import torch
import torchvision.transforms as transforms

from utils import get_device


class ImagePreprocess:
    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        resize_size: Tuple[int, int] = (256, 256),
    ) -> None:
        self.device = get_device()
        self.to_tensor = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # 256, 256 in the case of pixel space editing
        # 64, 64 in the case of latent space editing
        self.resize = transforms.Resize(resize_size)

    def process(self, image: torch.Tensor) -> torch.Tensor:
        return self.to_tensor(image).to(self.device)

    def resize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        mask = self.resize(mask.unsqueeze(0))
        mask[mask < 1] = 0
        mask[mask > 1] = 1
        mask = mask.squeeze()
        return mask
