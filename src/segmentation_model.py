import torch
from torchvision import transforms
from Segmentation.model import BiSeNet

from utils import get_device


class SegmentationModel:
    def __init__(
        self,
        ckpt: str = "Segmentation/res/cp/79999_iter.pth",
        n_classes: int = 19,
        image_size: tuple = (512, 512),
    ) -> None:
        self.device = get_device()
        self.net = self.load_net(ckpt, n_classes)

        self.to_tensor = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def process(self, image: torch.Tensor) -> torch.Tensor:
        return self.to_tensor(image).to(self.device)

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
        t_image = self.process(image)
        out = self.net(t_image)[0]
        segmentation = out.squeeze(0).argmax(0)

        return segmentation
