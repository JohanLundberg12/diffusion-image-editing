from typing import List
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from transforms import tensors_to_pils


def display_result_images_alongside_source_image(
    result_images: List[Image.Image], source_image: Image.Image
) -> Image.Image:
    images_to_concatenate = [np.array(source_image)] + [
        np.array(img) for img in result_images
    ]

    res = np.concatenate(images_to_concatenate, axis=1)

    return Image.fromarray(res)


def display_alongside_source_image(
    result_image: Image.Image, source_image: Image.Image
):
    res = np.concatenate(
        [
            np.array(source_image),
            np.array(result_image),
        ],
        axis=1,
    )
    return Image.fromarray(res)


def _to_np_image(tensor_img: torch.Tensor):
    # 1CHW -> HWC
    tensor_img = (
        (tensor_img.permute(0, 2, 3, 1) * 127.5 + 128)
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
        .numpy()[0]
    )
    return tensor_img


def show_tensor_img(tensor_img: torch.Tensor):
    np_img = _to_np_image(tensor_img)
    plt.imshow(np_img)
    plt.axis("off")


from typing import Union, List, Optional
from itertools import zip_longest


def get_num_rows(num_images: int, num_cols: int) -> int:
    """
    Calculates the number of rows needed to display a given number of images in a grid with a given number of columns.

    Args:
        num_images (int): The total number of images to display.
        num_cols (int): The number of columns in the grid.

    Returns:
        int: The number of rows needed to display all the images in the grid.
    """
    num_rows, num_cols = divmod(num_images, num_cols)  # return x // y, x % y

    if num_cols > 0:
        num_rows += 1

    return num_rows


def show_images_in_a_grid(
    images: List[Image.Image],
    num_cols: int = 1,
    titles: list = [],
    super_title: Optional[str] = None,
    figsize: tuple = (10, 10),
    y_labels=None,
):
    num_rows = get_num_rows(len(images), num_cols)
    fig, axarr = plt.subplots(num_rows, num_cols, figsize=figsize)

    for i, (img, title) in enumerate(zip_longest(images, titles, fillvalue="")):
        row = i // num_cols
        col = i % num_cols
        ax = axarr[row, col] if num_rows > 1 and num_cols > 1 else axarr[max(row, col)]
        ax.imshow(img, cmap="viridis")
        ax.set_title(title)

        if col == 0 and y_labels is not None:
            ax.set_ylabel(f"loss scale: {y_labels[row]}")
        else:
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

    image_pils = tensors_to_pils(samples)

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
