from typing import Optional

import torch

from image_preprocess import ImagePreprocess
from DDIMSegmentation.model import BiSeNet
from Morphology import Dilation2d


class MaskHandler:
    def __init__(
        self,
        preprocess: ImagePreprocess,
        net: BiSeNet,
        dilation_method: Optional[Dilation2d] = None,
    ) -> None:
        self.preprocess = preprocess
        self.net = net
        self.dilation_method = dilation_method
        self.segmentation: Optional[torch.Tensor] = None
        self.mask: Optional[torch.Tensor] = None

    def segment_image(self, image: torch.Tensor) -> torch.Tensor:
        t_image = self.preprocess.process(image)
        out = self.net(t_image)[0]
        self.segmentation = out.squeeze(0).argmax(0)
        return self.segmentation

    def create_mask(self, image: torch.Tensor, classes: list) -> None:
        segmentation = self.segment_image(image)
        masks = [
            self.create_class_mask(segmentation, class_label) for class_label in classes
        ]
        mask = sum(masks)
        mask = self.postprocess_mask(mask)  # type: ignore
        self.mask = mask

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

    def apply_mask(
        self, mask: torch.Tensor, zo: torch.Tensor, zv: torch.Tensor
    ) -> torch.Tensor:
        return mask * zv + ((1 - mask) * zo)

    def get_segmentation(self) -> torch.Tensor:
        if self.segmentation is not None:
            return self.segmentation.cpu()
        else:
            raise ValueError("Segmentation not available")

    def get_mask(self) -> torch.Tensor:
        if self.mask is not None:
            return self.mask.permute(0, 2, 3, 1).squeeze().cpu()
        else:
            raise ValueError("Mask not available.")
