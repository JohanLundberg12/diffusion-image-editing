# utils.py
from typing import Union, List, Optional
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from diffusers import DDIMPipeline
from itertools import zip_longest

from transforms import (
    reverse_transform,
    get_reverse_image_transform,
)


def get_device(verbose: bool = False) -> torch.device:
    """Returns the device to be used for training."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if verbose:
        print(f"Using {device} as backend")

    return device


def generate_random_samples(
    num_samples: int, model: DDIMPipeline
) -> Union[torch.Tensor, List[torch.Tensor]]:
    if num_samples == 1:
        return torch.randn(
            1,
            model.unet.config.in_channels,  # type: ignore
            model.unet.config.sample_size,  # type: ignore
            model.unet.config.sample_size,  # type: ignore
        ).to("cuda")
    else:
        return [
            generate_random_samples(1, model) for _ in range(num_samples)
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

    transform = get_reverse_image_transform()
    image_pils = [reverse_transform(sample, transform) for sample in samples]

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


# def imshow(img, title=""):
# img = img.to("cpu")
# img = img.permute(1, 2, 0, 3)
# img = img.reshape(img.shape[0], img.shape[1], -1)
# img = img / 2 + 0.5     # unnormalize
# img = torch.clamp(img, min=0., max=1.)
# npimg = img.numpy()
# plt.imshow(np.transpose(npimg, (1, 2, 0)))
# plt.title(title)
# plt.show()


def print_statistics(x_t: torch.Tensor, name: Optional[str] = None) -> None:
    if name is not None:
        print(f"Statistics for {name}")

    print(f"Mean of {name}: {x_t.mean():.2f}")
    print(f"Std of {name}: {x_t.std():.2f}")
    print(f"Min of {name}: {x_t.min():.2f}")
    print(f"Max of {name}: {x_t.max():.2f}")
