from abc import ABC, abstractmethod
from typing import Optional
import torch
from diffusers import DDIMPipeline  # type: ignore

from image_preprocess import ImagePreprocess
from DDIMSegmentation.model import BiSeNet


# l2 norm of x and y
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
    ) -> None:
        self.loss_scale = loss_scale
        self.mask: Optional[torch.Tensor] = None
        self.x_0 = None

    @property
    def name(self) -> str:
        """Return the name of the strategy class."""
        return self.__class__.__name__

    @abstractmethod
    def loss(self, p_t: torch.Tensor) -> torch.Tensor:
        """Calculate the loss."""
        pass

    def apply(
        self,
        input_image,
        model_output,
        step_time: int,
        pipeline: DDIMPipeline,
    ) -> torch.Tensor:
        """Apply the attribute function strategy to update the input image."""

        input_image = input_image.detach().requires_grad_(True)

        # p_t prediction (prediction of x_0)
        alpha_prod_t = pipeline.scheduler.alphas_cumprod[step_time]  # type: ignore
        beta_prod_t = 1 - alpha_prod_t
        p_t = (input_image - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (
            0.5
        )

        if self.mask is not None:
            attr_loss = self.loss(self.mask * p_t) + l2_norm(
                1 - self.mask * p_t, self.x_0
            )
        else:
            attr_loss = self.loss(p_t)
        attr_loss = attr_loss * self.loss_scale

        attr_grad = -torch.autograd.grad(attr_loss, input_image)[0]

        # if self.mask is not None:
        # attr_grad = attr_grad * self.mask

        input_image = input_image.detach() + attr_grad * alpha_prod_t**2

        return input_image


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
        net: BiSeNet,
        preprocess: ImagePreprocess,
        idx_for_class: list,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.net = net
        self.preprocess = preprocess
        self.idx_for_class = idx_for_class

    def loss(self, img):
        img = self.preprocess.process(img)
        out = self.net(img)[0]
        out = torch.nn.functional.softmax(out, dim=1)
        out = out.sum(dim=(2, 3)) / (512 * 512)
        out = out[0, self.idx_for_class]
        out = out.sum()

        return out
