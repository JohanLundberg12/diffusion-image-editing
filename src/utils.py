# utils.py
from typing import Union, List, Optional
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from diffusers import UNet2DModel
from itertools import zip_longest
from tqdm import tqdm

from transforms import tensor_to_pil


def apply_mask(
    mask: torch.Tensor,
    zo: Union[torch.Tensor, List[torch.Tensor]],
    zv: Union[torch.Tensor, List[torch.Tensor]],
) -> Union[torch.Tensor, List[torch.Tensor]]:
    if isinstance(zo, List) and isinstance(zv, List):
        return [mask * zv_i + ((1 - mask) * zo_i) for zv_i, zo_i in zip(zv, zo)]
    elif isinstance(zo, torch.Tensor) and isinstance(zv, torch.Tensor):
        return mask * zv + ((1 - mask) * zo)
    else:
        raise TypeError("zo and zv must be of the same type")


def get_device(verbose: bool = False) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"Using {device} as backend")

    return device


def generate_random_samples(
    num_samples: int, unet: UNet2DModel, generator: Optional[torch.Generator] = None
) -> Union[torch.Tensor, List[torch.Tensor]]:
    if num_samples == 1:
        return torch.randn(
            (
                1,
                unet.config.in_channels,  # type: ignore
                unet.config.sample_size,  # type: ignore
                unet.config.sample_size,  # type: ignore
            ),
            generator=generator,  # type: ignore
        ).to("cuda")
    else:
        return [
            generate_random_samples(1, unet, generator=generator)
            for _ in range(num_samples)
        ]  # type: ignore


def get_num_rows(num_images: int, num_cols: int) -> int:
    """
    Calculates the number of rows needed to display a given number of images in a grid with a given number of columns.

    Args:
        num_images (int): The total number of images to display.
        num_cols (int): The number of columns in the grid.

    Returns:
        int: The number of rows needed to display all the images in the grid.
    """
    num_rows = num_images // num_cols
    if num_images % num_cols > 0:
        num_rows += 1
    return num_rows


def show_images_in_a_grid(
    images: List[Image.Image],
    titles: list,
    num_cols: int,
    super_title: Optional[str] = None,
):
    num_rows = get_num_rows(len(images), num_cols)
    fig, axarr = plt.subplots(num_rows, num_cols)
    axarr = np.array(
        axarr
    )  # Make sure axarr is always an array, even when it's a single AxesSubplot object

    for i, img in enumerate(images):
        row = i // num_cols
        col = i % num_cols
        ax = axarr[row, col] if num_rows > 1 and num_cols > 1 else axarr[max(row, col)]
        ax.imshow(img, cmap="viridis")
        ax.set_title(titles[i])
        ax.axis("off")
    if super_title is not None:
        fig.suptitle(super_title)
    plt.tight_layout()


def display_samples(
    samples: Union[List[torch.Tensor], torch.Tensor],
    titles: List[str] = [],
    num_cols: int = 1,
    super_title: Optional[str] = None,
) -> None:
    """
    Displays the provided image samples.
    If more than one column is specified, displays the images in a grid.
    """
    if isinstance(samples, torch.Tensor):
        samples = [samples]

    image_pils = tensor_to_pil(samples)

    if num_cols > 1:
        show_images_in_a_grid(
            image_pils, titles=titles, num_cols=num_cols, super_title=super_title
        )
    else:
        if super_title:
            print(super_title)
            print("-" * len(super_title))  # add a separator line
        for title, img_pil in zip_longest(titles, image_pils, fillvalue=""):
            print(f"Title: {title}")
            plt.figure()
            display(img_pil)
            plt.show()


def create_progress_bar(steps: torch.Tensor | range, show_progbar: bool):
    enumerator = enumerate(steps)
    if show_progbar:
        return tqdm(enumerator)
    else:
        return enumerator
