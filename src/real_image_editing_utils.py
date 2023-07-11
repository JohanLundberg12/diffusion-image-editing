import dlib
import torch
from PIL import Image

from alignment import align_face
from diffusion import DiffusionModelFactory

from transforms import pil_to_tensor


def reverse_inverse_process(x_t, model, residuals=None, inversion=False):
    scheduler = model.scheduler
    timesteps = scheduler.timesteps
    residuals = list()

    if inversion:
        timesteps = reversed(timesteps)

    for step_idx, step_time in enumerate(timesteps):
        with torch.no_grad():
            residual = model.unet(x_t, step_time).sample

        if inversion:
            next_timestep = min(
                scheduler.config.num_train_timesteps - 2,
                step_time
                + scheduler.config.num_train_timesteps // scheduler.num_inference_steps,
            )
        else:
            prev_timestep = (
                step_time
                - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
            )

        # compute alphas, betas
        alpha_prod_t = scheduler.alphas_cumprod[step_time]
        if inversion:
            alpha_prod_t_next = (
                scheduler.alphas_cumprod[next_timestep]  # type: ignore
                if next_timestep >= 0  # type: ignore
                else scheduler.final_alpha_cumprod
            )
        else:
            alpha_prod_t_prev = (
                scheduler.alphas_cumprod[prev_timestep]  # type: ignore
                if prev_timestep >= 0  # type: ignore
                else scheduler.final_alpha_cumprod
            )

        beta_prod_t = 1 - alpha_prod_t

        # Compute predicted x_0
        p_t = (x_t - beta_prod_t ** (0.5) * residual) / alpha_prod_t ** (0.5)

        if scheduler.config.clip_sample:
            p_t = p_t.clamp(
                -scheduler.config.clip_sample_range, scheduler.config.clip_sample_range
            )

        # pred sample direction
        if inversion:
            pred_sample_direction = (1 - alpha_prod_t_next) ** (0.5) * residual  # type: ignore
        else:
            pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * residual  # type: ignore

        if inversion:
            x_t = alpha_prod_t_next ** (0.5) * p_t + pred_sample_direction  # type: ignore
        else:
            x_t = alpha_prod_t_prev ** (0.5) * p_t + pred_sample_direction  # type: ignore

        residuals.append(residual)

    return x_t, residuals


def compare_reverse_inverse_process_to_pipeline_synthesis(x_t: torch.Tensor):
    factory = DiffusionModelFactory(
        device="cuda",
        timesteps=50,
    )
    pipeline = factory.create_model(name="ddpm")

    x_0_gen, _ = reverse_inverse_process(x_t, pipeline, inversion=False)
    x_0_syn, _ = pipeline.synthesize_image(x_t, eta=0.0)  # type: ignore
    x_0_syn = pil_to_tensor(x_0_syn).to("cuda")  # type: ignore

    x_t_gen, _ = reverse_inverse_process(x_0_gen, pipeline, inversion=True)
    x_t_syn, _ = reverse_inverse_process(x_0_syn, pipeline, inversion=True)

    x_0_gen_gen, _ = reverse_inverse_process(x_t_gen, pipeline, inversion=False)
    x_0_gen_syn, _ = reverse_inverse_process(x_t_syn, pipeline, inversion=False)

    x_0_syn_gen, _ = pipeline.synthesize_image(x_t_gen, eta=0)  # type: ignore
    x_0_syn_syn, _ = pipeline.synthesize_image(x_t_syn, eta=0)  # type: ignore
    x_0_syn_gen = pil_to_tensor(x_0_syn_gen).to("cuda")  # type: ignore
    x_0_syn_syn = pil_to_tensor(x_0_syn_syn).to("cuda")  # type: ignore

    return x_0_gen, x_0_syn, x_0_gen_gen, x_0_gen_syn, x_0_syn_gen, x_0_syn_syn


def run_alignment(image_path: str) -> Image.Image:
    model = "../shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(model)  # type: ignore
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


def prepare_real_image_for_editing(
    image_path: str,
) -> torch.Tensor:
    img_aligned = run_alignment(image_path)
    img_aligned_rgb = img_aligned.convert("RGB")

    img = pil_to_tensor(img_aligned_rgb).to("cuda")  # type: ignore

    img = (img - img.mean()) / img.std()

    return img
