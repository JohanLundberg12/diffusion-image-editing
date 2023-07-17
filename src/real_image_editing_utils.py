import dlib
import torch
from PIL import Image

from diffusion_model_factory import DiffusionModelFactory

from alignment import align_face
from transforms import pil_to_tensor


def compare_reverse_inverse_process_to_pipeline_synthesis(x_t: torch.Tensor):
    factory = DiffusionModelFactory()
    pipeline = factory.create_model(name="ddpm")

    x_0_gen, _ = pipeline.reverse_inverse_process(x_t, pipeline, inversion=False)
    x_0_syn, _, _ = pipeline.generate_image(
        x_t, eta=0.0, return_pred_original_samples=False
    )
    x_0_syn = pil_to_tensor(x_0_syn).to("cuda")  # type: ignore

    x_t_gen, _ = pipeline.reverse_inverse_process(x_0_gen, pipeline, inversion=True)
    x_t_syn, _ = pipeline.reverse_inverse_process(x_0_syn, pipeline, inversion=True)

    x_0_gen_gen, _ = pipeline.reverse_inverse_process(
        x_t_gen, pipeline, inversion=False
    )
    x_0_gen_syn, _ = pipeline.reverse_inverse_process(
        x_t_syn, pipeline, inversion=False
    )

    x_0_syn_gen, _, _ = pipeline.generate_image(
        x_t_gen, eta=0, return_pred_original_samples=False
    )
    x_0_syn_syn, _, _ = pipeline.generate_image(
        x_t_syn, eta=0, return_pred_original_samples=False
    )
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
