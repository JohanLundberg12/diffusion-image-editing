import torch
from torchvision import transforms

from Morphology import Dilation2d
from utils import get_device


class MaskCreator:
    def __init__(
        self,
        dilate_mask: bool = True,
        resize_size: tuple = (256, 256),
    ) -> None:
        self.device = get_device()
        self.dilation_method = (
            Dilation2d(1, 1, 7, soft_max=False).to(self.device) if dilate_mask else None
        )
        # 256, 256 in the case of pixel space editing
        # 64, 64 in the case of latent space editing
        self.resize = transforms.Resize(resize_size)

    def create_mask(self, segmentation: torch.Tensor, classes: list) -> torch.Tensor:
        masks = [
            self.create_class_mask(segmentation, class_label) for class_label in classes
        ]
        mask = sum(masks)
        mask = self.postprocess_mask(mask)  # type: ignore

        return mask

    def create_class_mask(
        self, parsing: torch.Tensor, class_label: int
    ) -> torch.Tensor:
        mask = (parsing == class_label).float()

        if self.dilation_method:
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = self.dilation_method(mask)
            mask = mask.detach().cpu().permute(0, 2, 3, 1)
            mask = mask.squeeze().squeeze()
        return mask

    def postprocess_mask(self, mask: torch.Tensor) -> torch.Tensor:
        mask = self.resize_mask(mask)
        mask = torch.cat(
            [mask.unsqueeze(0), mask.unsqueeze(0), mask.unsqueeze(0)]
        ).unsqueeze(0)
        return mask.to(self.device)

    def resize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        mask = self.resize(mask.unsqueeze(0))
        mask[mask < 1] = 0
        mask[mask > 1] = 1
        mask = mask.squeeze()
        return mask
