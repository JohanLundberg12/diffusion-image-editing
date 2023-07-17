from typing import Optional, Union, Tuple, List

import torch

from PIL import Image

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
    UNet2DModel,
    VQModel,
)

from transforms import tensor_to_pil, tensors_to_pils
from utils import create_progress_bar, get_device, generate_random_samples, set_seed


def get_alpha_prod_t(
    alphas_cumprod: torch.Tensor, timestep: torch.Tensor
) -> torch.Tensor:
    alpha_prod_t = alphas_cumprod[timestep]
    return alpha_prod_t


def pred_original_samples(img, alpha_prod_t, model_output) -> torch.Tensor:
    beta_prod_t = 1 - alpha_prod_t
    p_t = (img - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    return p_t


class BaseDiffusion:
    def __init__(
        self,
        unet: Union[UNet2DModel, UNet2DConditionModel],
        scheduler: DDIMScheduler,
        vae: Optional[Union[VQModel, AutoencoderKL]] = None,
    ) -> None:
        self.device = get_device()
        self.unet = unet
        self.scheduler = scheduler
        self.vae = vae
        self.data_dimensionality = self.unet.sample_size

    def reverse_inverse_process(
        self,
        x_t: torch.Tensor,
        inversion=False,
        prompt="",
        guidance_scale=7.5,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        scheduler = self.scheduler
        timesteps = scheduler.timesteps
        model_outputs = list()

        additional_setup = self._additional_setup(prompt, guidance_scale)

        if inversion:
            timesteps = reversed(timesteps)

        for step_idx, step_time in enumerate(timesteps):
            with torch.no_grad():
                model_output = self.predict_model_output(
                    x_t,
                    step_time,
                    **additional_setup,
                ).sample

            if inversion:
                next_timestep = min(
                    scheduler.config.num_train_timesteps - 2,
                    step_time
                    + scheduler.config.num_train_timesteps
                    // scheduler.num_inference_steps,
                )
            else:
                prev_timestep = (
                    step_time
                    - scheduler.config.num_train_timesteps
                    // scheduler.num_inference_steps
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

            p_t = pred_original_samples(x_t, alpha_prod_t, model_output)

            if scheduler.config.clip_sample:
                p_t = p_t.clamp(
                    -scheduler.config.clip_sample_range,
                    scheduler.config.clip_sample_range,
                )

            # pred sample direction
            if inversion:
                pred_sample_direction = (1 - alpha_prod_t_next) ** (0.5) * model_output  # type: ignore
            else:
                pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * model_output  # type: ignore

            if inversion:
                x_t = alpha_prod_t_next ** (0.5) * p_t + pred_sample_direction  # type: ignore
            else:
                x_t = alpha_prod_t_prev ** (0.5) * p_t + pred_sample_direction  # type: ignore

            model_outputs.append(model_output)

        return x_t, model_outputs

    def _setup_generation(
        self,
        num_inference_steps: int,
        xt: torch.Tensor,
        eta: float,
        zs: List[torch.Tensor] | None,
        generator: torch.Generator,
    ) -> Tuple[torch.Tensor, Union[List[torch.Tensor], None]]:
        self.scheduler.set_timesteps(num_inference_steps)

        if xt is None:
            xt = generate_random_samples(1, self.unet, generator=generator)  # type: ignore
        if zs is None and eta > 0:
            zs = generate_random_samples(  # type: ignore
                num_inference_steps, self.unet, generator=generator  # type: ignore
            )

        return xt, zs

    def _additional_setup(self, prompt: str = "", guidance_scale: float = 7.5) -> dict:
        return {}

    def encode(self, sample: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            return sample
        else:
            raise NotImplementedError

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            return latent
        else:
            raise NotImplementedError

    def predict_model_output(
        self, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_variance_noise(
        self, zs: List[torch.Tensor], step_idx: int, eta: float
    ) -> torch.Tensor | None:
        return zs[step_idx] if eta > 0 else None

    def single_step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        xt: torch.Tensor,
        eta: float,
        variance_noise: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.scheduler.step(
            model_output, timestep, xt, eta=eta, variance_noise=variance_noise  # type: ignore
        )
        xt, pred_original_sample = output.to_tuple()  # type: ignore

        return xt, pred_original_sample

    def diffusion_loop(
        self, xt, eta, zs, pbar, **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        new_model_outputs = list()
        pred_original_samples = list()

        for step_idx, timestep in pbar:
            model_output = self.predict_model_output(xt, timestep, **kwargs)

            variance_noise = self.get_variance_noise(zs, step_idx, eta)

            xt, pred_original_sample = self.single_step(
                model_output=model_output,
                timestep=timestep,
                xt=xt,
                eta=eta,
                variance_noise=variance_noise,  # type: ignore
            )

            new_model_outputs.append(model_output)
            pred_original_samples.append(pred_original_sample.detach())

        sample = self.decode(xt)

        pred_original_samples = torch.stack(
            pred_original_samples, dim=0
        ).squeeze()  # B, 1, C, H, W -> B, C, H, W

        pred_original_samples = [
            self.decode(pred_original_sample.unsqueeze(0))  # C, H, W -> 1, C, H, W
            for pred_original_sample in pred_original_samples
        ]

        return sample, new_model_outputs, pred_original_samples

    def generate_image(
        self,
        xt: Optional[torch.Tensor] = None,
        eta: float = 0,
        zs: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        show_progbar: bool = True,
        return_pred_original_samples: bool = True,
        generator: Optional[torch.Generator] = None,
        prompt: str = "",
        guidance_scale: float = 7.5,
    ) -> Tuple[Image.Image, List[torch.Tensor], Optional[List[Image.Image]]]:
        if generator is None:
            generator = set_seed(seed)

        xt, zs = self._setup_generation(  # type: ignore
            num_inference_steps=num_inference_steps,
            xt=xt,  # type: ignore
            eta=eta,
            zs=zs,  # type: ignore
            generator=generator,
        )
        # additional setup specific to the child class
        additional_setup = self._additional_setup(
            prompt=prompt, guidance_scale=guidance_scale
        )
        pbar = create_progress_bar(self.scheduler.timesteps, show_progbar)

        sample, new_model_outputs, pred_original_samples = self.diffusion_loop(
            xt=xt,
            eta=eta,
            zs=zs,
            pbar=pbar,
            **additional_setup,  # pass additional parameters to diffusion_loop
        )

        img = tensor_to_pil(sample)

        if return_pred_original_samples:
            pred_original_samples = tensors_to_pils(pred_original_samples)
            return img, new_model_outputs, pred_original_samples
        else:
            return img, new_model_outputs, None  # type: ignore

    def generate_images(
        self,
        num_images: int = 1,
        num_inference_steps: int = 50,
        eta: float = 0,
        seed: Optional[int] = None,
        show_progbar: bool = True,
        **kwargs,
    ) -> Tuple[
        List[List[Image.Image]], List[List[torch.Tensor]], List[List[Image.Image]]
    ]:
        generator = set_seed(seed)

        all_imgs = []
        all_model_outputs = []
        all_original_sample_preds = []

        pbar = create_progress_bar(range(num_images), show_progbar)

        for i in pbar:
            imgs, model_outputs, original_sample_preds = self.generate_image(
                num_inference_steps=num_inference_steps,
                eta=eta,
                seed=seed,
                show_progbar=show_progbar,
                generator=generator,
                **kwargs,
            )
            all_imgs.append(imgs)
            all_model_outputs.append(model_outputs)
            all_original_sample_preds.append(original_sample_preds)

        return all_imgs, all_model_outputs, all_original_sample_preds
