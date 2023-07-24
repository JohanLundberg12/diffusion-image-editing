from typing import List, Optional, Tuple
import torch


def calculate_variance(model, timestep):
    prev_timestep = get_previous_timestep(model, timestep)
    alpha_prod_t, alpha_prod_t_prev = compute_alpha_products(
        model, timestep, prev_timestep
    )
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance


def compute_alpha_products(model, timestep, prev_timestep):
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        model.scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else model.scheduler.final_alpha_cumprod
    )
    return alpha_prod_t, alpha_prod_t_prev


def compute_predicted_original_sample(sample, beta_prod_t, model_output, alpha_prod_t):
    """3. compute predicted original sample from predicted noise also called
    "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    """
    return (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)


def tokenize_text(model, prompt):
    """Tokenize prompt for conditional diffusion."""
    text_input = model.tokenizer(
        [prompt],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_input


def encode_text(model, prompts):
    """Encode prompt for conditional diffusion."""
    text_input = tokenize_text(model, prompts)

    with torch.no_grad():
        text_encoding = model.text_encoder(text_input.input_ids.to(model.device))[0]
    return text_encoding


def get_noise_pred(
    model, latent, t, text_emb: Optional[torch.Tensor] = None, cfg_scale: float = 3.5
):
    """Return the predicted noise for a given latent and timestep."""
    with torch.no_grad():
        if text_emb is not None:
            latents_input = torch.cat([latent] * 2)
            noise_pred = model.unet(
                sample=latents_input,
                timestep=t,
                encoder_hidden_states=text_emb,
                cfg_scale=cfg_scale,
            )["sample"]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg_scale * (
                noise_pred_text - noise_pred_uncond
            )
        else:
            noise_pred = model.unet(latent, t)["sample"]
    return noise_pred


def get_previous_timestep(model, timestep):
    return (
        timestep
        - model.scheduler.config.num_train_timesteps
        // model.scheduler.num_inference_steps
    )


def get_variance_noise(
    zs: torch.Tensor, step_idx: int, eta: float
) -> torch.Tensor | None:
    return zs[step_idx] if zs is not None and eta != 0 else None


def single_step(
    model,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    sample: torch.Tensor,
    eta: float,
    variance_noise: torch.Tensor | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scheduler_output = model.scheduler.step(
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        eta=eta,
        variance_noise=variance_noise,
    )
    xt, pred_original_sample = scheduler_output.to_tuple()

    return xt, pred_original_sample


def diffusion_loop(
    model, xt: torch.Tensor, eta: float, zs: torch.Tensor, text_emb=None, cfg_scale=3.5
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    new_model_outputs = list()
    pred_original_samples = list()

    for step_idx, timestep in enumerate(model.scheduler.timesteps):
        noise_pred = get_noise_pred(model, xt, timestep, text_emb, cfg_scale)

        variance_noise = get_variance_noise(zs, step_idx, eta)

        xt, pred_original_sample = single_step(
            model,
            model_output=noise_pred,
            timestep=timestep,
            sample=xt,
            eta=eta,
            variance_noise=variance_noise,
        )

        new_model_outputs.append(noise_pred)
        pred_original_samples.append(pred_original_sample.detach())

    return xt, new_model_outputs, pred_original_samples


def prep_text(model, prompt: str) -> torch.Tensor:
    # add unconditional embedding
    return torch.cat([encode_text(model, ""), encode_text(model, prompt)])