from typing import Optional
import torch
from tqdm import tqdm

from diffusion_utils import (
    compute_alpha_products,
    compute_predicted_original_sample,
    encode_text,
    get_noise_pred,
    get_previous_timestep,
    calculate_variance,
    get_variance_noise,
)


def mu_tilde(model, xt, x0, timestep):
    "mu_tilde(x_t, x_0) DDPM paper eq. 7"
    prev_timestep = get_previous_timestep(model, timestep)

    alpha_prod_t, alpha_prod_t_prev = compute_alpha_products(
        model, timestep, prev_timestep
    )

    beta_t = 1 - alpha_prod_t
    alpha_bar = model.scheduler.alphas_cumprod[timestep]
    return ((alpha_prod_t_prev**0.5 * beta_t) / (1 - alpha_bar)) * x0 + (
        (alpha_prod_t**0.5 * (1 - alpha_prod_t_prev)) / (1 - alpha_bar)
    ) * xt


def sample_xts_from_x0(model, x0, num_inference_steps=50):
    """
    Forward diffusion sampling: P(x_1:T|x_0)
    """
    alpha_bar = model.scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
    variance_noise_shape = (
        num_inference_steps,
        model.unet.in_channels,
        model.unet.sample_size,
        model.unet.sample_size,
    )

    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xts = torch.zeros(variance_noise_shape).to(x0.device)
    for t in reversed(timesteps):
        idx = t_to_idx[int(t)]
        xts[idx] = (
            x0 * (alpha_bar[t] ** 0.5)
            + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]
        )
    xts = torch.cat([xts, x0], dim=0)

    return xts


def forward_step(model, model_output, timestep, sample):
    next_timestep = min(
        model.scheduler.config.num_train_timesteps - 2,
        timestep
        + model.scheduler.config.num_train_timesteps
        // model.scheduler.num_inference_steps,
    )

    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = compute_predicted_original_sample(
        sample, beta_prod_t, model_output, alpha_prod_t
    )

    next_sample = model.scheduler.add_noise(
        pred_original_sample, model_output, torch.LongTensor([next_timestep])
    )
    return next_sample


def inversion_forward_process(
    model,
    x0,
    etas=None,
    num_inference_steps=50,
    prompt: Optional[str] = None,
    cfg_scale: float = 3.5,
    prog_bar=False,
):
    if prompt is not None:
        text_emb = encode_text(model, prompt)
        uncond_emb = encode_text(model, "")
        context = torch.cat([text_emb, uncond_emb])
    else:
        context = None

    model.scheduler.set_timesteps(num_inference_steps)
    timesteps = model.scheduler.timesteps.to(model.device)
    variance_noise_shape = (
        num_inference_steps,
        model.unet.in_channels,
        model.unet.sample_size,
        model.unet.sample_size,
    )
    if etas is None or (type(etas) in [int, float] and etas == 0):
        eta_is_zero = True
        zs = None
    else:
        eta_is_zero = False
        if type(etas) in [int, float]:
            etas = [etas] * model.scheduler.num_inference_steps
        xts = sample_xts_from_x0(model, x0, num_inference_steps=num_inference_steps)
        alpha_bar = model.scheduler.alphas_cumprod
        zs = torch.zeros(size=variance_noise_shape, device=model.device)

    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xt = x0
    rev_timesteps = reversed(timesteps)
    op = tqdm(rev_timesteps) if prog_bar else rev_timesteps

    for t in op:
        idx = t_to_idx[int(t)]

        # 1. predict noise residual
        if not eta_is_zero:
            xt = xts[idx][None]  # increase shape to (1, C, H, W)

        noise_pred = get_noise_pred(model, xt, t, context, cfg_scale)

        if eta_is_zero:
            # 2. compute more noisy image and set x_t -> x_t+1
            xt = forward_step(model, noise_pred, t, xt)

        else:
            xtm1 = xts[idx + 1][None]  # x_{t-1}

            # pred of x0: P(f_t(x_t))
            beta_prod_t = 1 - alpha_bar[t]
            pred_original_sample = compute_predicted_original_sample(
                xt, beta_prod_t, noise_pred, alpha_bar[t]
            )

            prev_timestep = get_previous_timestep(model, t)

            alpha_prod_t_prev = (
                model.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else model.scheduler.final_alpha_cumprod
            )

            variance = calculate_variance(model, t)

            # Direction pointing to x_t: D(f_t(x_t))
            pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance) ** (
                0.5
            ) * noise_pred

            # mu_hat(x_t)
            mu_xt = (
                alpha_prod_t_prev ** (0.5) * pred_original_sample
                + pred_sample_direction
            )

            # z_t equation 5 in https://inbarhub.github.io/DDPM_inversion/
            z = (xtm1 - mu_xt) / (etas[idx] * variance**0.5)
            zs[idx] = z

            # correction to avoid error accumulation (eq. 3 in https://inbarhub.github.io/DDPM_inversion/)
            xtm1 = mu_xt + (etas[idx] * variance**0.5) * z
            xts[idx + 1] = xtm1

    if not zs is None:
        zs[-1] = torch.zeros_like(zs[-1])

    if eta_is_zero:
        xts = None
    return xt, zs, xts


def invert(
    model,
    x0: torch.FloatTensor,
    num_inference_steps=50,
    eta=1,
    prompt: Optional[str] = None,
    cfg_scale: float = 3.5,
    prog_bar=True,
):
    #  inverts a real image according to Algorihm 1 in https://arxiv.org/pdf/2304.06140.pdf,
    #  based on the code in https://github.com/inbarhub/DDPM_inversion

    #  returns wt, zs, wts:
    #  xt - inverted latent
    #  xts - intermediate inverted latents
    #  zs - noise maps associated with the image

    # find xt, zs and xts - forward process
    xt, zs, xts = inversion_forward_process(
        model, x0, num_inference_steps=num_inference_steps, etas=eta, prompt=prompt, cfg_scale=cfg_scale, prog_bar=True
    )
    return xt, zs, xts


def reverse_step(model, model_output, timestep, sample, eta=0, variance_noise=None):
    # 1. get previous step value (=t-1)
    prev_timestep = get_previous_timestep(model, timestep)

    # 2. compute alphas, betas
    alpha_prod_t, alpha_prod_t_prev = compute_alpha_products(
        model, timestep, prev_timestep
    )
    beta_prod_t = 1 - alpha_prod_t
    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (
        sample - beta_prod_t ** (0.5) * model_output
    ) / alpha_prod_t ** (0.5)
    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    # variance = self.scheduler._get_variance(timestep, prev_timestep)
    variance = calculate_variance(model, timestep)  # , prev_timestep)
    std_dev_t = eta * variance ** (0.5)
    # Take care of asymetric reverse process (asyrp)
    model_output_direction = model_output
    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output_direction
    pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (
        0.5
    ) * model_output_direction
    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = (
        alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    )
    # 8. Add noice if eta > 0
    if eta > 0:
        if variance_noise is None:
            variance_noise = torch.randn(model_output.shape, device=model.device)
        sigma_z = eta * variance ** (0.5) * variance_noise
        prev_sample = prev_sample + sigma_z

    return prev_sample


def inversion_reverse_process(
    model,
    xT,
    eta=0,
    zs=None,
    prompt: Optional[str] = None,
    cfg_scale: float = 3.5,
    prog_bar=False,
):
    if prompt is not None:
        context = encode_text(model, prompt)
    else:
        context = None

    timesteps = model.scheduler.timesteps.to(model.device)

    xt = xT.expand(1, -1, -1, -1)  # C x H x W -> 1 x C x H x W
    op = tqdm(timesteps[-zs.shape[0] :]) if prog_bar else timesteps[-zs.shape[0] :]

    timestep_to_idx = {
        int(timestep): idx for idx, timestep in enumerate(timesteps[-zs.shape[0] :])
    }

    for t in op:
        idx = timestep_to_idx[int(t)]

        # 1. predict noise residual
        noise_pred = get_noise_pred(model, xt, t, context, cfg_scale)

        z = get_variance_noise(zs, idx, eta)

        # 2. compute less noisy image and set x_t -> x_t-1
        xt = reverse_step(model, noise_pred, t, xt, eta=eta, variance_noise=z)

    return xt, zs


def sample(
    model,
    zs,
    xts,
    Tskip=36,
    eta=1,
    prompt: Optional[str] = None,
    cfg_scale: float = 3.5,
    prog_bar=True,
):
    """Sample image using DDPM inversion from https://inbarhub.github.io/DDPM_inversion/.
    Args:
        model: DDPM model
        zs: noise maps associated with the image
        xts: intermediate inverted latents
        Tskip: Starting generation process from wts[Tskip] or (T - Tskip) (in the paper Tskip = 36).
            Larger -> more adherence to the original image.
        eta: noise level.
    """

    # reverse process (via Zs and wT)
    x0, zs = inversion_reverse_process(
        model,
        xT=xts[Tskip],
        eta=eta,
        zs=zs[Tskip:],
        prompt=prompt,
        cfg_scale=cfg_scale,
        prog_bar=prog_bar,
    )

    if x0.dim() < 4:
        x0 = x0[None, :, :, :]
    return x0
