from collections import defaultdict
import torch

from SegDiffEditPipeline import SegDiffEditPipeline


from models import get_pretrained_anyGAN
from transforms import pil_to_tensor
from utils import generate_random_samples

from constants import ANY_GAN_ATTRS_DICT


def lpips(original: torch.Tensor, edited: torch.Tensor):
    """
    Calculates the LPIPS between the original and edited image.

    Args:
        original (torch.Tensor): The original image.
        edited (torch.Tensor): The edited image.

    Returns:
        float: The LPIPS between the original and edited image.
    """
    loss_fn_vgg = lpips.LPIPS(net="vgg").to("cuda")
    loss = loss_fn_vgg(original, edited)
    return loss


# possible TODO: add support for different attr_funcs and other parameters.
def avg_increase_decrease_per_attribute(
    editor: SegDiffEditPipeline, diffusion_model, attr_func, n_samples, generator
):
    f"""
    Computes the average increase/decrease in the anyGAN model prediction for each attribute.
    This is of shape (1, 40, 2). The first dimension is the batch dimension, the second is the
    attribute dimension, and the third is the prediction dimension. The prediction dimension
    is 0 for the zeroth class and 1 for the first class. The attribute dimension is the index
    of the attribute in the anyGAN model. The increase/decrease
    is computed by taking the difference between the original and edited image's anyGAN prediction
    for each attribute. Each of these are of shape (1, 40, 2).
    This is then sorted along the attribute dimension.

    Args:
        editor (SegDiffEditPipeline): The editor to use.
        diffusion_model (BaseDiffusion): The diffusion model to use.
        attr_func (Callable): The attribute function to use.
        n_samples (int): The number of samples to evaluate on.
        generator (torch.Generator): The generator to use for random sampling.

    Returns:
        Tuple[Dict[str, float], Dict[str, float]]: A tuple of two dictionaries. The first
        dictionary is the average increase/decrease in the anyGAN model prediction for every attribute along
        prediction dimension 0.
        The second dictionary is the average increase/decrease in the anyGAN model prediction for every attribute along
        prediction dimension 1.

    Example return:
        Predictions for anyGAN prediction dimension 0:
            {'39 Young: 1.34441020488739',
            '18 Heavy_Makeup: 1.2829089760780334',
            '36 Wearing_Lipstick: 1.0046115458011626',
            '2 Attractive: 0.9877155601978302',}
        Predictions for anyGAN prediction dimension 1:
            {'20 Male: 1.0743066191673278',
            '17 Gray_Hair: 0.8982777118682861',
            '10 Blurry: 0.7925063610076905',
            '14 Double_Chin: 0.7053128719329834',}

    """
    predictor = get_pretrained_anyGAN()
    predictor.eval()

    ANY_GAN_ATTRS_DICT_REV = {v: k for k, v in ANY_GAN_ATTRS_DICT.items()}

    d_zero = defaultdict(float)
    d_one = defaultdict(float)

    for _ in range(n_samples):
        xt = generate_random_samples(1, diffusion_model.unet, generator=generator)
        zs = generate_random_samples(50, diffusion_model.unet, generator=generator)

        img, model_outputs, pred_original_samples = diffusion_model.generate_image(
            xt=xt,
            eta=1,
            zs=zs,
            num_of_inference_steps=50,
            show_progbar=True,
            return_pred_original_samples=True,
        )

        img_t = pil_to_tensor(img).to("cuda")
        o_attr = predictor(img_t).view(-1, 40, 2)

        img_edit = editor.edit_image(
            img=img,
            xt=xt,
            model_outputs=model_outputs,
            eta=1,
            zs=zs,
            attr_func=attr_func,
        )[0]

        img_edit_t = pil_to_tensor(img_edit).to("cuda")

        edit_attr = predictor(img_edit_t).view(-1, 40, 2)

        # get difference between original and edited
        diff = edit_attr - o_attr

        # sort each column by difference
        differences, indexes = torch.sort(diff, dim=1, descending=True)

        # create dict of attr_idx attr_name: difference for each column
        attr_pred_index_zero_diff = {
            f"{i.item()} {ANY_GAN_ATTRS_DICT_REV[i.item()]}": j.item()
            for i, j in zip(indexes[0, :, 0], differences[0, :, 0])
        }
        attr_pred_index_one_diff = {
            f"{i.item()} {ANY_GAN_ATTRS_DICT_REV[i.item()]}": j.item()
            for i, j in zip(indexes[0, :, 1], differences[0, :, 1])
        }

        # update both dicts
        for k, v in attr_pred_index_zero_diff.items():
            d_zero[k] += v
        for k, v in attr_pred_index_one_diff.items():
            d_one[k] += v

    # divide by n_samples to get average difference
    d_zero = {k: v / n_samples for k, v in d_zero.items()}
    d_one = {k: v / n_samples for k, v in d_one.items()}

    return d_zero, d_one


# to get results of d_zero or d_one: sorted(list(d_zero.items()), key=lambda x: x[1], reverse=True)


# possible TODO: add support for different attr_funcs and other parameters.
def attribute_consistency(
    editor: SegDiffEditPipeline, diffusion_model, attr_func, n_samples, generator
):
    """
    Computes the attribute consistency for every anyGAN attribute on n_samples
    using the provided editor and diffusion model. Consistency is defined as the
    percentage of times the attribute is correctly predicted after editing with the
    initial predictions being the ground truth.

    Args:
        editor (SegDiffEditPipeline): The editor to use.
        diffusion_model (BaseDiffusion): The diffusion model to use.
        attr_func (Callable): The attribute function to use.
        n_samples (int): The number of samples to evaluate on.
        generator (torch.Generator): The generator to use for random sampling.

    Returns:
        Dict[str, float]: A dictionary of attribute name to consistency percentage.

    Example return:
        5_o_Clock_Shadow              100.00%
        Arched_Eyebrows               83.00%
        Attractive                    87.00%
        Bags_Under_Eyes               97.00%
        Bald                          100.00%
        ...
        Young                         87.00%
    """
    predictor = get_pretrained_anyGAN()
    predictor.eval()

    accs = 0
    for _ in range(n_samples):
        xt = generate_random_samples(1, diffusion_model.unet, generator=generator)
        zs = generate_random_samples(50, diffusion_model.unet, generator=generator)

        img, model_outputs, pred_original_samples = diffusion_model.generate_image(
            xt=xt,
            eta=1,
            zs=zs,
            num_of_inference_steps=50,
            show_progbar=True,
            return_pred_original_samples=True,
        )

        img_t = pil_to_tensor(img).to("cuda")
        o_attr = predictor(img_t).view(-1, 40, 2)

        img_edit = editor.edit_image(
            img=img,
            xt=xt,
            model_outputs=model_outputs,
            eta=1,
            zs=zs,
            attr_func=attr_func,
        )[0]

        img_edit_t = pil_to_tensor(img_edit).to("cuda")

        edit_attr = predictor(img_edit_t).view(-1, 40, 2)

        attr = torch.argmax(o_attr, dim=2)
        edit_attr = torch.argmax(edit_attr, dim=2)

        this_acc = (attr == edit_attr).float().mean(0)
        accs = accs + this_acc
    accs = accs / n_samples

    return accs
