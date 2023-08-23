from typing import Optional, Union
import torch

from diffusers import (
    DDIMScheduler,
    UNet2DConditionModel,
    UNet2DModel,
)

from diffusion_utils import (
    diffusion_loop,
    single_step,
    get_variance_noise,
    get_noise_pred,
)

from transforms import tensor_to_pil

from utils import (
    create_progress_bar,
    get_device,
    initialize_random_samples,
    set_seed,
    generate_random_samples,
    process_lists_of_tensors,
)


class Diffusion:
    def __init__(
        self,
        model,
    ) -> None:
        self.device = get_device()
        self.model = model
        self.unet: Union[UNet2DConditionModel, UNet2DModel] = self.model.unet
        self.scheduler: DDIMScheduler = self.model.scheduler
        self.data_dimensionality = self.unet.sample_size

    def encode(self, sample: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def additional_prep(self, model, prompt):
        return None

    def generate_image(
        self,
        xt,
        eta=0,
        zs=None,
        num_inference_steps=50,
        generator=None,
        prompt="",
        cfg_scale=3.5,
        return_xts=False,
    ):
        self.model.scheduler.set_timesteps(num_inference_steps)

        text_emb = self.additional_prep(self.model, prompt)

        model_outputs = list()
        pred_original_samples = list()
        xts = list()

        if eta > 0 and zs is None:
            zs = generate_random_samples(
                num_inference_steps, self.model.unet, generator=generator
            )
        for step_idx, timestep in diffusion_loop(model=self.model, zs=zs):
            noise_pred = get_noise_pred(self.model, xt, timestep, text_emb, cfg_scale)
            variance_noise = get_variance_noise(zs, step_idx, eta)

            xt, pred_original_sample = single_step(
                self.model, noise_pred, timestep, xt, eta, variance_noise
            )

            model_outputs.append(noise_pred)
            pred_original_samples.append(pred_original_sample.detach())

            if return_xts:
                xts.append(xt.detach())

        x0 = xt
        x0 = self.decode(x0)

        img = tensor_to_pil(x0)

        pred_original_samples = process_lists_of_tensors(self, pred_original_samples)

        if return_xts:
            xts = process_lists_of_tensors(self, xts)
            return img, model_outputs, pred_original_samples, xts
        else:
            return img, model_outputs, pred_original_samples, None

    def generate_images(
        self,
        num_images: int = 1,
        eta: float = 0,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        show_progbar: bool = True,
        return_pred_original_samples: bool = True,
        prompt: str = "",
        cfg_scale: float = 3.5,
        return_xts: bool = False,
    ):
        generator = set_seed(seed)

        self.scheduler.set_timesteps(num_inference_steps)

        all_xts = []
        all_zs = []
        all_imgs = []
        all_model_outputs = []
        all_original_sample_preds = []

        pbar = create_progress_bar(range(num_images), show_progbar)

        for i in pbar:
            xt, zs = initialize_random_samples(
                self.model,
                num_inference_steps=num_inference_steps,
                eta=eta,
                generator=generator,
            )
            all_xts.append(xt)
            all_zs.append(zs)
            sample, model_outputs, pred_original_samples, xts = self.generate_image(
                xt=xt,
                eta=eta,
                zs=zs,
                num_inference_steps=num_inference_steps,
                generator=generator,
                prompt=prompt,
                cfg_scale=cfg_scale,
                return_xts=return_xts,
            )
            all_imgs.append(sample)
            all_model_outputs.append(model_outputs)

            if return_pred_original_samples:
                all_original_sample_preds.append(pred_original_samples)

        return all_imgs, all_model_outputs, all_original_sample_preds, all_xts, all_zs
