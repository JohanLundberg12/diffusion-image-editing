import dlib
import torch
from PIL import Image

from alignment import align_face

from transforms import pil_to_tensor


def run_alignment(image_path: str) -> Image.Image:
    model = "../shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(model)  # type: ignore
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


def prepare_real_image_for_editing(
    image_path: str,
) -> torch.Tensor:
    img_aligned = run_alignment(image_path).convert("RGB")
    img = pil_to_tensor(img_aligned).to("cuda")

    return img
