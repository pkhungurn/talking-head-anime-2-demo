import math
import os
from typing import List

import PIL.Image
import numpy
import torch
from matplotlib import cm
from torch import Tensor


def is_power2(x):
    return x != 0 and ((x & (x - 1)) == 0)


def torch_save(content, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'wb') as f:
        torch.save(content, f)


def torch_load(file_name):
    with open(file_name, 'rb') as f:
        return torch.load(f)


def srgb_to_linear(x):
    x = numpy.clip(x, 0.0, 1.0)
    return numpy.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = numpy.clip(x, 0.0, 1.0)
    return numpy.where(x <= 0.003130804953560372, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)


def image_linear_to_srgb(image):
    assert image.shape[2] == 3 or image.shape[2] == 4
    if image.shape[2] == 3:
        return linear_to_srgb(image)
    else:
        height, width, _ = image.shape
        rgb_image = linear_to_srgb(image[:, :, 0:3])
        a_image = image[:, :, 3:4]
        return numpy.concatenate((rgb_image, a_image), axis=2)


def image_srgb_to_linear(image):
    assert image.shape[2] == 3 or image.shape[2] == 4
    if image.shape[2] == 3:
        return srgb_to_linear(image)
    else:
        height, width, _ = image.shape
        rgb_image = srgb_to_linear(image[:, :, 0:3])
        a_image = image[:, :, 3:4]
        return numpy.concatenate((rgb_image, a_image), axis=2)


def save_rng_state(file_name):
    rng_state = torch.get_rng_state()
    torch_save(rng_state, file_name)


def load_rng_state(file_name):
    rng_state = torch_load(file_name)
    torch.set_rng_state(rng_state)


def optimizer_to_device(optim, device):
    for state in optim.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def grid_change_to_numpy_image(torch_image, num_channels=3):
    height = torch_image.shape[1]
    width = torch_image.shape[2]
    size_image = (torch_image[0, :, :] ** 2 + torch_image[1, :, :] ** 2).sqrt().view(height, width, 1).numpy()
    hsv = cm.get_cmap('hsv')
    angle_image = hsv(((torch.atan2(
        torch_image[0, :, :].view(height * width),
        torch_image[1, :, :].view(height * width)).view(height, width) + math.pi) / (2 * math.pi)).numpy()) * 3
    numpy_image = size_image * angle_image[:, :, 0:3]
    rgb_image = linear_to_srgb(numpy_image)
    if num_channels == 3:
        return rgb_image
    elif num_channels == 4:
        return numpy.concatenate([rgb_image, numpy.ones_like(size_image)], axis=2)
    else:
        raise RuntimeError("Unsupported num_channels: " + str(num_channels))


def rgb_to_numpy_image(torch_image: Tensor, min_pixel_value=-1.0, max_pixel_value=1.0):
    assert torch_image.dim() == 3
    assert torch_image.shape[0] == 3
    height = torch_image.shape[1]
    width = torch_image.shape[2]

    reshaped_image = torch_image.numpy().reshape(3, height * width).transpose().reshape(height, width, 3)
    numpy_image = (reshaped_image - min_pixel_value) / (max_pixel_value - min_pixel_value)
    return linear_to_srgb(numpy_image)


def rgba_to_numpy_image_greenscreen(torch_image: Tensor,
                                    min_pixel_value=-1.0,
                                    max_pixel_value=1.0,
                                    include_alpha=False):
    height = torch_image.shape[1]
    width = torch_image.shape[2]

    numpy_image = (torch_image.numpy().reshape(4, height * width).transpose().reshape(height, width,
                                                                                      4) - min_pixel_value) \
                  / (max_pixel_value - min_pixel_value)
    rgb_image = linear_to_srgb(numpy_image[:, :, 0:3])
    a_image = numpy_image[:, :, 3]
    rgb_image[:, :, 0:3] = rgb_image[:, :, 0:3] * a_image.reshape(a_image.shape[0], a_image.shape[1], 1)
    rgb_image[:, :, 1] = rgb_image[:, :, 1] + (1 - a_image)

    if not include_alpha:
        return rgb_image
    else:
        return numpy.concatenate((rgb_image, numpy.ones_like(numpy_image[:, :, 3:4])), axis=2)


def rgba_to_numpy_image(torch_image: Tensor, min_pixel_value=-1.0, max_pixel_value=1.0):
    assert torch_image.dim() == 3
    assert torch_image.shape[0] == 4
    height = torch_image.shape[1]
    width = torch_image.shape[2]

    reshaped_image = torch_image.numpy().reshape(4, height * width).transpose().reshape(height, width, 4)
    numpy_image = (reshaped_image - min_pixel_value) / (max_pixel_value - min_pixel_value)
    rgb_image = linear_to_srgb(numpy_image[:, :, 0:3])
    a_image = numpy_image[:, :, 3]
    rgba_image = numpy.concatenate((rgb_image, a_image.reshape(height, width, 1)), axis=2)
    return rgba_image


def extract_PIL_image_from_filelike(file):
    return PIL.Image.open(file)


def extract_pytorch_image_from_filelike(file, scale=2.0, offset=-1.0):
    return extract_pytorch_image_from_PIL_image(extract_PIL_image_from_filelike(file), scale, offset)


def extract_pytorch_image_from_PIL_image(pil_image, scale=2.0, offset=-1.0):
    image = extract_numpy_image_from_PIL_image(pil_image)
    h, w, c = image.shape
    image = image.reshape(h * w, c)
    for pixel in image:
        if pixel[3] == 0.0:
            pixel[0:3] = 0.0
    image = image.transpose().reshape(c, h, w)
    torch_image = torch.from_numpy(image).float() * scale + offset
    return torch_image


def extract_numpy_image_from_filelike(file):
    return extract_numpy_image_from_PIL_image(extract_PIL_image_from_filelike(file))


def extract_numpy_image_from_PIL_image(pil_image):
    image = numpy.asarray(pil_image) / 255.0
    image[:, :, 0:3] = srgb_to_linear(image[:, :, 0:3])
    return image


def create_parent_dir(file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)


def run_command(command_parts: List[str]):
    command = " ".join(command_parts)
    os.system(command)


def resize_PIL_image(pil_image, size=(256, 256)):
    w, h = pil_image.size
    d = min(w, h)
    r = ((w - d) // 2, (h - d) // 2, (w + d) // 2, (h + d) // 2)
    return pil_image.resize(size, resample=PIL.Image.LANCZOS, box=r)


def save_pytorch_image(image, file_name):
    if image.shape[0] == 1:
        image = image.squeeze()
    if image.shape[0] == 4:
        numpy_image = rgba_to_numpy_image(image.detach().cpu())
        pil_image = PIL.Image.fromarray(numpy.uint8(numpy.rint(numpy_image * 255.0)), mode='RGBA')
    else:
        numpy_image = rgb_to_numpy_image(image.detach().cpu())
        pil_image = PIL.Image.fromarray(numpy.uint8(numpy.rint(numpy_image * 255.0)), mode='RGB')
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    pil_image.save(file_name)


def convert_output_image_from_torch_to_numpy(output_image):
    if output_image.shape[2] == 2:
        h, w, c = output_image.shape
        output_image = torch.transpose(output_image.reshape(h * w, c), 0, 1).reshape(c, h, w)
    if output_image.shape[0] == 4:
        numpy_image = rgba_to_numpy_image(output_image)
    elif output_image.shape[0] == 1:
        c, h, w = output_image.shape
        alpha_image = torch.cat([output_image.repeat(3, 1, 1) * 2.0 - 1.0, torch.ones(1, h, w)], dim=0)
        numpy_image = rgba_to_numpy_image(alpha_image)
    elif output_image.shape[0] == 2:
        numpy_image = grid_change_to_numpy_image(output_image, num_channels=4)
    else:
        raise RuntimeError("Unsupported # image channels: %d" % output_image.shape[0])
    return numpy_image
