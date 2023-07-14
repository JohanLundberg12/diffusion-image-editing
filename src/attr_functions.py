from abc import ABC, abstractmethod
import lpips
import torch
from torchvision import models

from segmentation_model import SegmentationModel

from utils import get_alpha_prod_t, pred_original_samples


def l2_norm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculate the l2 norm between two tensors."""
    return torch.sqrt(torch.sum((x - y) ** 2))


def single_color_loss(images: torch.Tensor, idx: int, target: float) -> torch.Tensor:
    """Calculate the mean absolute error between a single color channel and its target."""
    error = torch.abs(images[:, idx, :, :] - target).mean()
    return error


def color_loss(
    images: torch.Tensor, r_target: float, g_target: float, b_target: float
) -> torch.Tensor:
    """Calculate the weighted mean absolute error between each color channel and their targets."""
    r_error = single_color_loss(images, 0, r_target)
    g_error = single_color_loss(images, 1, g_target)
    b_error = single_color_loss(images, 2, b_target)
    color_loss = r_error * r_target + g_error * g_target + b_error * b_target

    return color_loss


class AttrFunc(ABC):
    """Abstract base class for different attribute function strategies."""

    def __init__(
        self,
        loss_scale: float = 1.0,
        stop_at=None,
    ) -> None:
        self.loss_scale = loss_scale
        self.stop_at = stop_at

    @property
    def name(self) -> str:
        """Return the name of the strategy class."""
        return self.__class__.__name__

    @abstractmethod
    def loss(self, p_t: torch.Tensor, **kwargs) -> torch.Tensor:
        """Calculate the loss."""
        pass

    def apply(
        self,
        input_image,
        pred_x_0,
        model_output,
        timestep,
        step_idx,
        scheduler,
        mask=None,
        x_0=None,
    ) -> torch.Tensor:
        """Apply the attribute function strategy to update the input image."""

        input_image = input_image.detach().requires_grad_(True)

        alpha_prod_t = get_alpha_prod_t(scheduler.alphas_cumprod, timestep)
        p_t = pred_original_samples(input_image, alpha_prod_t, model_output)

        if self.stop_at is not None and step_idx >= self.stop_at:
            return input_image, p_t

        # if mask is not None and x_0 is not None:
        #    attr_loss = self.loss(mask * p_t) + l2_norm(1 - mask * p_t, x_0)
        # else:
        #    attr_loss = self.loss(p_t)
        attr_loss = self.loss(p_t)
        attr_loss = attr_loss * self.loss_scale
        if step_idx % 10 == 0:
            print(step_idx, "loss:", attr_loss.item())

        attr_grad = -torch.autograd.grad(attr_loss, input_image)[0]

        if mask is not None:
            attr_grad = mask * attr_grad

        input_image = input_image.detach() + attr_grad * alpha_prod_t**2

        return input_image, p_t


class SingleColorAttrFunc(AttrFunc):
    """Attribute function strategy for a single color channel."""

    def __init__(self, target: float, color_idx: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.target = target
        self.color_idx = color_idx

    def loss(self, p_t: torch.Tensor) -> torch.Tensor:
        return single_color_loss(p_t, self.color_idx, self.target)


class MultiColorAttrFunc(AttrFunc):
    """Attribute function strategy for multiple color channels."""

    def __init__(
        self,
        r_target: float,
        g_target: float,
        b_target: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.r_target = r_target
        self.g_target = g_target
        self.b_target = b_target

    def loss(self, p_t: torch.Tensor) -> torch.Tensor:
        return color_loss(
            p_t,
            r_target=self.r_target,
            g_target=self.g_target,
            b_target=self.b_target,
        )


class NetAttrFunc(AttrFunc):
    def __init__(
        self,
        segmentation_model: SegmentationModel,
        idx_for_class: list,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.segmentation_model = segmentation_model
        self.idx_for_class = idx_for_class

    def loss(self, img, i):
        out = self.segmentation_model(img)[0]
        out = torch.nn.functional.softmax(out, dim=1)
        out = out.sum(dim=(2, 3)) / (512 * 512)
        out = out[0, self.idx_for_class]
        out = out.sum()

        return out


def _load_model(model: torch.nn.Module, state_dict_path: str) -> torch.nn.Module:
    """Load a model from a state dict."""

    # load model weights
    state_dict = torch.load(state_dict_path, map_location="cuda")["state_dict"]

    # load weights to model
    model.load_state_dict(state_dict)

    return model


def _get_pretrained_anyGAN():
    # manual download of pretrained models:
    # URL = "https://hanlab.mit.edu/projects/anycost-gan/files/attribute_predictor.pt"

    predictor = models.resnet50()
    predictor.fc = torch.nn.Linear(predictor.fc.in_features, 40 * 2)
    predictor = _load_model(predictor, "../attribute_predictor.pt")

    return predictor.to("cuda")


class AnyGANAttrFunc(AttrFunc):
    def __init__(self, idx_for_class, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = _get_pretrained_anyGAN()
        self.idx_for_class = idx_for_class

    def loss(self, img):
        attr = self.model(img).view(-1, 40, 2)
        attr_max_preds = torch.argmax(attr, dim=2)

        attr_max_value_idx = attr_max_preds[0][self.idx_for_class]
        max_class_value = attr[0][self.idx_for_class][0][attr_max_value_idx]

        return max_class_value
