import torch

from image_preprocess import ImagePreprocess
from Morphology import Dilation2d


class MaskCreator:
    def __init__(
        self,
        preprocess: ImagePreprocess,
        dilate_mask: bool = True,
    ) -> None:
        self.preprocess = preprocess
        self.dilation_method = (
            Dilation2d(1, 1, 7, soft_max=False).to("cuda") if dilate_mask else None
        )

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
        mask = self.preprocess.resize_mask(mask)
        mask = torch.cat(
            [mask.unsqueeze(0), mask.unsqueeze(0), mask.unsqueeze(0)]
        ).unsqueeze(0)
        return mask.to("cuda")
