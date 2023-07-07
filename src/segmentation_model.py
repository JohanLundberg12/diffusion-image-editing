import torch

from image_preprocessor import ImagePreprocessor
from Segmentation.model import BiSeNet

from utils import get_device


class SegmentationModel:
    def __init__(
        self,
        image_preprocessor: ImagePreprocessor,
        ckpt: str = "Segmentation/res/cp/79999_iter.pth",
        n_classes: int = 19,
    ) -> None:
        self.net = self.load_net(ckpt, n_classes)
        self.device = get_device()
        self.image_preprocessor = image_preprocessor

    def load_net(
        self,
        ckpt: str,
        n_classes: int,
    ) -> BiSeNet:
        net = BiSeNet(n_classes=n_classes)
        net.to(self.device)

        net.load_state_dict(torch.load(ckpt))
        net.eval()

        return net

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        t_image = self.image_preprocessor.process(image)
        out = self.net(t_image)[0]
        segmentation = out.squeeze(0).argmax(0)

        return segmentation
