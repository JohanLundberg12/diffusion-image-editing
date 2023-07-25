# utils.py
from typing import List, Optional
import torch
from diffusers import UNet2DModel, UNet2DConditionModel
from PIL import Image
from tqdm import tqdm

from transforms import tensor_to_pil


def to_pil_and_decode_batch_of_tensors(
    model, tensor: torch.Tensor
) -> List[Image.Image]:
    return [tensor_to_pil(model.decode(t.unsqueeze(0))) for t in tensor]


# move to helper functions or make it part of the decode call?
def process_lists_of_tensors(model, tensors: List[torch.Tensor]) -> List[Image.Image]:
    tensor_stacked = torch.stack(tensors, dim=0).squeeze()
    return to_pil_and_decode_batch_of_tensors(model, tensor_stacked)


def apply_mask(
    mask: torch.Tensor,
    zo: torch.Tensor,
    zv: torch.Tensor,
) -> torch.Tensor:
    return mask * zv + ((1 - mask) * zo)


def get_device(verbose: bool = False) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"Using {device} as backend")

    return device


def initialize_random_samples(
    model,
    num_inference_steps: int,
    eta: float,
    generator: torch.Generator,
):
    xt = generate_random_samples(1, model.unet, generator=generator)

    if eta > 0:
        zs = generate_random_samples(
            num_inference_steps, model.unet, generator=generator
        )
    else:
        zs = None

    return xt, zs


def generate_random_samples(
    num_samples: int | None,
    unet: UNet2DModel | UNet2DConditionModel,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if num_samples is None:
        num_samples = 1

    return torch.randn(
        (
            num_samples,
            unet.config.in_channels,  # type: ignore
            unet.config.sample_size,  # type: ignore
            unet.config.sample_size,  # type: ignore
        ),
        generator=generator,  # type: ignore
    ).to("cuda")


def create_progress_bar(steps: torch.Tensor | range, show_progbar: bool):
    enumerator = enumerate(steps)
    if show_progbar:
        return tqdm(enumerator, total=len(steps))
    else:
        return enumerator


def set_seed(seed: int | None) -> torch.Generator:
    if seed is None:
        seed = int(torch.randint(int(1e6), (1,)))
    return torch.manual_seed(seed)


# something like this I could use:
# directions = ){
#        'eye_openness':            (54,  7,  8,  20),
#        'smile':                   (46,  4,  5, -20),
#        'trimmed_beard':           (58,  7,  9,  20),
#        'white_hair':              (57,  7, 10, -24),
#        'lipstick':                (34, 10, 11,  20)
#    }
# editor.apply_ganspace(latents, ganspace_pca, [directions["white_hair"], directions["eye_openness"], directions["smile"]])
