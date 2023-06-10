# attr_func_strategy.py
from abc import ABC, abstractmethod

import torch


def blue_loss(images):
    error = torch.abs(images[:, -1, :, :] - 0.9).mean()
    return error


class AttrFuncStrategy(ABC):
    @abstractmethod
    def apply(
        self, input_image, residual, step_time, model, mask=None, mask_attr=False
    ):
        pass


class BlueAttrFuncStrategy(AttrFuncStrategy):
    def apply(
        self,
        input_image,
        residual,
        step_time,
        model,
        mask=None,
        mask_attr=False,
        blueiness=1.0,
    ):
        input_image = torch.tensor(
            input_image.detach().cpu().numpy(),
            requires_grad=True,
            device=input_image.device,
        )

        # p_t prediction (prediction of x_0)
        alpha_prod_t = model.scheduler.alphas_cumprod[step_time]
        beta_prod_t = 1 - alpha_prod_t
        p_t = (input_image - beta_prod_t ** (0.5) * residual) / alpha_prod_t ** (0.5)

        blue_loss_scale = blueiness
        attr = blue_loss(p_t) * blue_loss_scale

        attr_grad = -torch.autograd.grad(attr, input_image)[0]

        if mask_attr and mask is not None:
            attr_grad = attr_grad * mask

        input_image = input_image.detach() + attr_grad * alpha_prod_t**2

        return input_image
