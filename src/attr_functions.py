from abc import ABC, abstractmethod
from typing import List

import lpips
import torch

from models import SegmentationModel

from diffusion_utils import compute_predicted_original_sample


def l2_norm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculate the l2 norm between two tensors."""
    return torch.sqrt(torch.sum((x - y) ** 2))


def apply_lpips(xt, x0, loss_fn_vgg):
    loss = loss_fn_vgg(xt, x0)

    return loss


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


def _init_loss_scales(loss_scales: torch.Tensor, t1: int, t2: int) -> torch.Tensor:
    num_elements = loss_scales.size().numel()
    interval_len = t2 - t1

    if num_elements != interval_len:
        if num_elements < interval_len and num_elements == 1:
            loss_scales = loss_scales.repeat(interval_len)
        else:
            raise ValueError(
                f"loss_scales must be of length {interval_len} or 1, but got {num_elements}"
            )

    return loss_scales


class AttrFunc(ABC):
    """Abstract base class for different attribute function strategies."""

    def __init__(
        self, loss_scale: float = 1, t1: int = 0, t2: int = 50, **kwargs
    ) -> None:
        self.kwargs = kwargs
        self.loss_scale = torch.Tensor(loss_scale).to("cuda")
        self.t1 = t1
        self.t2 = t2

        # prepare lpips metric
        if kwargs.get("use_lpips", False):
            self.loss_fn_vgg = lpips.LPIPS(net="vgg")
            self.metric = apply_lpips
        elif kwargs.get("use_l2", False):
            self.metric = l2_norm
        else:
            pass

    @property
    def name(self) -> str:
        """Return the name of the strategy class."""
        return self.__class__.__name__

    @abstractmethod
    def loss(self, pred_original_sample: torch.Tensor, **kwargs) -> torch.Tensor:
        """Calculate the loss."""
        raise NotImplementedError

    def calculate_loss(
        self, pred_original_sample: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Calculate the loss and do something with it."""
        if kwargs.get("mask_pred_original_sample", False):
            lambda_ = kwargs.get("lambda")
            metric = kwargs.get("metric")
            mask = kwargs.get("mask")
            x_0 = kwargs.get("x_0")

            assert lambda_ is not None
            assert metric is not None
            assert mask is not None
            assert x_0 is not None

            if kwargs.get("use_lpips", False):
                attr_loss = self.loss(mask * pred_original_sample) + lambda_ * metric(
                    1 - mask * pred_original_sample, x_0, self.loss_fn_vgg
                )
            elif kwargs.get("use_l2", False):
                attr_loss = self.loss(mask * pred_original_sample) + lambda_ * metric(
                    1 - mask * pred_original_sample, x_0
                )
            else:
                raise ValueError("No metric specified")
        else:
            attr_loss = self.loss(pred_original_sample, **kwargs)

        return attr_loss

    def edit_attr_grad(self, attr_grad, **kwargs):
        if kwargs.get("mask_attr_grad", False):
            if kwargs.get("mask", False) is None:
                raise ValueError("No mask specified")
            attr_grad = kwargs.get("mask") * attr_grad

        return attr_grad

    def get_attr_grad(self, xt, pred_original_sample, loss_scale, **kwargs):
        attr_loss = self.calculate_loss(pred_original_sample, **kwargs)
        attr_loss = attr_loss * loss_scale
        attr_grad = -torch.autograd.grad(attr_loss, xt)[0]
        attr_grad = self.edit_attr_grad(attr_grad, **kwargs)

        return attr_grad

    def apply(
        self,
        xt,  # must have parameter
        model_output,  # must have parameter
        timestep,  # must have parameter
        step_idx,  # must have parameter
        model,  # must have parameter
        **kwargs,
    ) -> torch.Tensor:
        """Apply the attribute function strategy to update the input image.
        Kwargs could be:
            - use_mask: to use a mask
            - mask: torch.Tensor (mask to apply to the pred_original_sample or to the attribute gradient)
            - x_0: torch.Tensor (original x0)
            - mask_attr_grad: bool (whether to mask the attr_grad)
            - metric: callable (how to regularize the loss)
            - lambda: float (weight of the regularization)

        """

        # don't apply attr func if we are outside the interval
        if step_idx < self.t1 or step_idx >= self.t2:
            return xt
        else:
            loss_scale = self.loss_scale

        xt = xt.detach().requires_grad_(True)
        alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = compute_predicted_original_sample(
            xt, beta_prod_t, model_output, alpha_prod_t
        )
        pred_original_sample = model.decode(pred_original_sample, no_grad=False)

        attr_grad = self.get_attr_grad(xt, pred_original_sample, loss_scale, **kwargs)

        xt = xt.detach() + attr_grad * alpha_prod_t**2

        return xt


class SingleColorAttrFunc(AttrFunc):
    """Attribute function strategy for a single color channel."""

    def __init__(self, target: float, color_idx: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.target = target
        self.color_idx = color_idx

    def loss(self, p_t: torch.Tensor, **kwargs) -> torch.Tensor:
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
        self.norm_factor = None

    def loss(self, img, **kwargs):
        out = self.segmentation_model.net(img)[0]  # 1, 19, 256, 256

        # get cls out
        # out_cls = out[0, self.idx_for_class]  # 256, 256
        # out_softmax = torch.nn.functional.softmax(out_cls, dim=1)  # 256, 256
        # out_sum = out_softmax.sum(dim=(0, 1)) / (256 * 256)  # 19
        # val = out_sum.sum()

        # other approach
        out_softmax = torch.nn.functional.softmax(out, dim=1)  # 1, 19, 256, 256
        out_sum = out_softmax.sum(dim=(2, 3)) / (256 * 256)  # 1, 19
        out_cls = out_sum[0, self.idx_for_class]

        if self.norm_factor is None:
            # Compute the normalization factor as the sum of the mask values
            mask = torch.any(
                torch.stack(
                    [(torch.argmax(out, dim=1) == idx) for idx in self.idx_for_class]
                ),
                dim=0,
            ).float()

            # Compute the normalization factor as the sum of the mask values
            self.norm_factor = mask.sum()

        # If norm_factor is 0, it means that there are no pixels for this class. Avoid dividing by zero.
        if self.norm_factor > 0:
            val = out_cls.sum() / self.norm_factor
        else:
            val = out_cls.sum()

        print("val: ", val)

        return val


class AnyGANAttrFunc(AttrFunc):
    def __init__(self, predictor, idx_for_class, **kwargs) -> None:
        super().__init__(**kwargs)
        self.predictor = predictor
        self.idx_for_class = idx_for_class

    def loss(self, xt, **kwargs):
        attr = self.predictor(xt).view(-1, 40, 2)
        max_class_value = attr[0][self.idx_for_class].max()

        return max_class_value
