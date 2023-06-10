# utils.py
from typing import TYPE_CHECKING, List
from typing import Union
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


def generate_random_samples(num_samples, model):
    if num_samples == 1:
        return torch.randn(
            1,
            model.unet.config.in_channels,
            model.unet.config.sample_size,
            model.unet.config.sample_size,
        ).to("cuda")
    else:
        return [generate_random_samples(1, model) for _ in range(num_samples)]


def masking(mask, zo, zv):
    return mask * zv + ((1 - mask) * zo)


def synthesize_residual(
    model, input_image, step_idx, step_time, mask=None, residuals_o: List = []
):
    with torch.no_grad():
        residual = model.unet(input_image, step_time).sample

        if mask is not None:
            residual = masking(mask, residuals_o[step_idx], residual)

    return residual


def dilation_mask(mask, dilation2d):
    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = dilation2d(mask).detach().cpu().permute(0, 2, 3, 1).squeeze().squeeze()

    return mask


def create_mask(parsing, classes, dilation, preprocess, dilation2d) -> torch.Tensor:
    masks = list()
    for cls in classes:
        mask = parsing == cls
        mask = mask.float()
        if dilation:
            mask = dilation_mask(mask, dilation2d)
        masks.append(mask)
    mask = sum(masks)
    mask = preprocess.resize_mask(mask)
    mask = torch.cat(
        [mask.unsqueeze(0), mask.unsqueeze(0), mask.unsqueeze(0)]
    ).unsqueeze(0)
    mask = mask.to("cuda")

    return mask


def get_num_rows(num_images: int, num_cols: int) -> int:
    num_rows = num_images // num_cols
    if num_images % num_cols > 0:
        num_rows += 1
    return num_rows


def show_images_in_a_grid(images: list, titles: list, num_cols: int):
    num_rows = get_num_rows(len(images), num_cols)
    fig, axarr = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    for i in range(len(images)):
        if num_rows == 1:
            axarr[i].imshow(images[i], cmap="gray")
            axarr[i].set_title(titles[i])
            axarr[i].axis("off")
        elif num_cols == 1:
            axarr[i].imshow(images[i], cmap="gray")
            axarr[i].set_title(titles[i])
            axarr[i].axis("off")
        else:
            row = i // num_cols
            col = i % num_cols
            axarr[row, col].imshow(images[i], cmap="gray")
            axarr[row, col].set_title(titles[i])
            axarr[row, col].axis("off")
    plt.tight_layout()


def display_samples(samples, titles, num_cols):
    images = list()

    for sample in samples:
        image_processed = sample.cpu().permute(0, 2, 3, 1)
        image_processed = (image_processed + 1.0) * 127.5
        image_processed = image_processed.numpy().astype(np.uint8)
        image_pil = Image.fromarray(image_processed[0])
        images.append(image_pil)

    show_images_in_a_grid(images, titles, num_cols)


def swap_axes(array: np.ndarray) -> np.ndarray:
    """
    Swaps the axes of a numpy array.
    """
    if array.ndim == 3 and (array.shape[0] == 3 or array.shape[0] == 1):
        array = np.transpose(
            array, (1, 2, 0)
        )  # you could also do array.swapaxes(0,2).swapaxes(0,1)
    return array


def display_sample(sample: Union[np.ndarray, torch.Tensor], imshow=False):
    """
    Displays a single image sample.
    """
    if torch.is_tensor(sample):
        if TYPE_CHECKING:
            assert isinstance(sample, torch.Tensor)
        if sample.device.type != "cpu":
            sample = sample.cpu()
        if sample.ndim == 4:
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.squeeze()
        sample = sample.numpy()

    if TYPE_CHECKING:
        assert isinstance(sample, np.ndarray)

    sample_min = np.min(sample)
    sample_max = np.max(sample)
    sample_normalized = (sample - sample_min) / (sample_max - sample_min) * 255
    sample_normalized = sample_normalized.astype(np.uint8)

    if imshow:
        return plt.imshow(sample_normalized)
    else:
        image_pil = Image.fromarray(sample_normalized)

        return display(image_pil)


# plot three images as a 1 x 3 grid
def plot_images(images, titles, suptitle=""):
    # if an image is a tensor on cuda convert it to numpy and ndim =4
    images = [
        ((image.cpu().permute(0, 2, 3, 1).numpy().squeeze() + 1.0) * 127.5).astype(
            np.uint8
        )
        if ((isinstance(image, torch.Tensor)) and (image.ndim == 4))
        else image
        for image in images
    ]

    fig, axs = plt.subplots(1, len(images), figsize=(10, 3))
    for i, (image, title) in enumerate(zip(images, titles)):
        axs[i].imshow(image)
        axs[i].set_title(title)
        axs[i].axis("off")
    fig.suptitle(suptitle)
    plt.show()
