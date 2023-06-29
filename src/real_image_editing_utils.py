from typing import Callable, List, Optional, Union
import dlib
import torch
from PIL import Image

from alignment import align_face
from diffusion_synthesizer import (
    DiffusionSynthesizer,
    PixelSpacePipeline,
    LatentPipeline,
    reverse_inverse_process,
)

from transforms import (
    get_image_transform,
    image_transform,
)


def compare_reverse_inverse_process_to_pipeline_synthesis(
    x_t: torch.Tensor, pipeline: Union[LatentPipeline, PixelSpacePipeline]
):
    image_synthesizer = DiffusionSynthesizer(pipeline)

    x_0_gen, _ = reverse_inverse_process(x_t, pipeline, inversion=False)
    x_0_syn, _ = image_synthesizer.synthesize_image(x_t, eta=0.0)

    x_t_gen, _ = reverse_inverse_process(x_0_gen, pipeline, inversion=True)
    x_t_syn, _ = reverse_inverse_process(x_0_syn, pipeline, inversion=True)

    x_0_gen_gen, _ = reverse_inverse_process(x_t_gen, pipeline, inversion=False)
    x_0_gen_syn, _ = reverse_inverse_process(x_t_syn, pipeline, inversion=False)

    x_0_syn_gen, _ = image_synthesizer.synthesize_image(x_t_gen, eta=0)
    x_0_syn_syn, _ = image_synthesizer.synthesize_image(x_t_syn, eta=0)

    return x_0_gen, x_0_syn, x_0_gen_gen, x_0_gen_syn, x_0_syn_gen, x_0_syn_syn


def run_alignment(image_path: str) -> Image.Image:
    predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


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

    return img
