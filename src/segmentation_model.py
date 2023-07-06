import torch

from image_preprocess import ImagePreprocess
from Segmentation.model import BiSeNet


class SegmentationModel:
    def __init__(self, net: BiSeNet, preprocess: ImagePreprocess) -> None:
        self.net = net
        self.preprocess = preprocess

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        t_image = self.preprocess.process(image)
        out = self.net(t_image)[0]
        segmentation = out.squeeze(0).argmax(0)

        return segmentation
