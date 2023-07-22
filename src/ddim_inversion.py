# Code: https://inbarhub.github.io/DDPM_inversion/

from typing import Optional

from typing import Union
import numpy as np
import torch
from tqdm import tqdm

from diffusion_utils import encode_text, get_noise_pred


def next_step(
    model,
    model_output: Union[torch.FloatTensor, np.ndarray],
    timestep: int,
    sample: Union[torch.FloatTensor, np.ndarray],
):
    """Prediction of next sample x_{t+1} from x_t, t, and the model output."""
    timestep, next_timestep = (
        min(
            timestep
            - model.scheduler.config.num_train_timesteps
            // model.scheduler.num_inference_steps,
            999,
        ),
        timestep,
    )

    # Compute alpha and beta products for current and next timestep
    alpha_prod_t = (
        model.scheduler.alphas_cumprod[timestep]
        if timestep >= 0
        else model.scheduler.final_alpha_cumprod
    )
    alpha_prod_t_sqrt = alpha_prod_t**0.5
    beta_prod_t_sqrt = (1 - alpha_prod_t) ** 0.5
    alpha_prod_t_next = model.scheduler.alphas_cumprod[next_timestep]
    alpha_prod_t_next_sqrt = alpha_prod_t_next**0.5

    next_original_sample = (
        sample - beta_prod_t_sqrt * model_output
    ) / alpha_prod_t_sqrt
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output

    next_sample = alpha_prod_t_next_sqrt * next_original_sample + next_sample_direction

    return next_sample


@torch.no_grad()
def ddim_loop(model, latent, prompt: Optional[str] = None, cfg_scale=3.5):
    """x_T <- x_0 via DDIM inversion."""
    if prompt is not None:
        text_emb = encode_text(model, prompt)
        uncond_emb = encode_text(model, "")
        context = torch.cat([text_emb, uncond_emb])
    else:
        context = None

    for i in tqdm(range(model.scheduler.num_inference_steps)):
        t = model.scheduler.timesteps[len(model.scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred(
            model, latent, t, text_emb=context, cfg_scale=cfg_scale
        )
        latent = next_step(model, noise_pred, t, latent)

    return latent


@torch.no_grad()
def ddim_inversion(model, x0, prompt: Optional[str] = None, cfg_scale: float = 3.5):
    latent = x0.clone().detach()
    xT = ddim_loop(model, latent, prompt=prompt, cfg_scale=cfg_scale)
    return xT
