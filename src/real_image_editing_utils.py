from typing import Callable, List, Optional, Union
import dlib
import numpy as np
import torch
from PIL import Image
from diffusers import DDIMPipeline
from torchvision import transforms
from torchvision.transforms import Lambda

from alignment import align_face
from image_synthesizer import ImageSynthesizer


# should I take residuals as a parameter in case I want to regenerate a particular image?
def reverse_inverse_process(
    x_t: torch.Tensor, model: DDIMPipeline, inversion: bool = False
) -> Union[torch.Tensor, List[torch.Tensor]]:
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
                step_time + scheduler.config.num_train_timesteps // scheduler.num_,
            )
            (
                step_time
                + scheduler.config.num_train_timesteps // scheduler.num_inference_steps
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
                scheduler.alphas_cumprod[next_timestep]
                if next_timestep >= 0
                else scheduler.final_alpha_cumprod
            )
        else:
            alpha_prod_t_prev = (
                scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
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
            pred_sample_direction = (1 - alpha_prod_t_next) ** (0.5) * residual
        else:
            pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * residual

        if inversion:
            x_t = alpha_prod_t_next ** (0.5) * p_t + pred_sample_direction
        else:
            x_t = alpha_prod_t_prev ** (0.5) * p_t + pred_sample_direction

        residuals.append(residual)

    return x_t, residuals


def compare_reverse_inverse_process_to_model_synthesis(
    x_t: torch.Tensor, model: DDIMPipeline
):
    image_synthesizer = ImageSynthesizer(model)

    x_0_gen, _ = reverse_inverse_process(x_t, model, inversion=False)
    x_0_syn, _ = image_synthesizer.synthesize_image(x_t, eta=0.0)

    x_t_gen, _ = reverse_inverse_process(x_0_gen, model, inversion=True)
    x_t_syn, _ = reverse_inverse_process(x_0_syn, model, inversion=True)

    x_0_gen_gen, _ = reverse_inverse_process(x_t_gen, model, inversion=False)
    x_0_gen_syn, _ = reverse_inverse_process(x_t_syn, model, inversion=False)

    x_0_syn_gen, _ = image_synthesizer.synthesize_image(x_t_gen, eta=0)
    x_0_syn_syn, _ = image_synthesizer.synthesize_image(x_t_syn, eta=0)

    return x_0_gen, x_0_syn, x_0_gen_gen, x_0_gen_syn, x_0_syn_gen, x_0_syn_syn


def run_alignment(image_path: str) -> Image.Image:
    predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


def input_t(input: torch.Tensor) -> torch.Tensor:
    # Scale between [-1, 1]
    return 2 * input - 1


def get_image_transform(image_size: int) -> Callable[[Image.Image], torch.Tensor]:
    """Returns a transform that scales the image pixel values to [-1, 1]"""

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # Shape: HWC, Scales data into [0,1] by div / 255, so does normalization
            transforms.Lambda(input_t),
        ]
    )

    return transform


def output_t(input: torch.Tensor) -> torch.Tensor:
    # Scale between [0, 1]
    return (input + 1) / 2


def get_reverse_image_transform() -> Callable[[torch.Tensor], Image.Image]:
    """Returns a transform that scales the image pixel values to [0, 255]
    and converts it to a PIL image."""

    reverse_transform = transforms.Compose(
        [
            Lambda(output_t),
            Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            Lambda(lambda t: t * 255.0),  # back to 0-255
            Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ]
    )
    return reverse_transform


def image_transform(
    image: Image.Image, transform: Callable[[Image.Image], torch.Tensor]
) -> torch.Tensor:
    """Transforms an image to a tensor and scales its pixel values to be between -1 and 1."""

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Reshape to (B, C, H, W)

    return image_tensor


def reverse_transform(
    image_tensor: torch.Tensor, transform: Callable[[torch.Tensor], Image.Image]
) -> Image.Image:
    """Transforms a tensor to an image and scales its pixel values to be between 0 and 255."""

    assert (
        len(image_tensor.shape) == 4
    )  # Expected image tensor to have 4 dimensions (B, C, H, W)

    image_tensor = image_tensor.squeeze(0)  # Reshape to (C, H, W)

    image = transform(image_tensor)

    return image


def prepare_real_image_for_editing(
    image_path, processing_steps: Optional[List[Callable]] = None
) -> torch.Tensor:
    img_aligned = run_alignment(image_path)
    img_aligned_rgb = img_aligned.convert("RGB")

    transform = get_image_transform(256)
    img = image_transform(img_aligned_rgb, transform).to("cuda")

    img = (img - img.mean()) / img.std()

    if processing_steps is not None:
        for step in processing_steps:
            img = step(img)
    # img = img.clamp(-1, 1)

    return img
