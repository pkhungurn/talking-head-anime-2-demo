import torch
from torch import Tensor
from torch.nn.functional import affine_grid, grid_sample


def apply_rgb_change(alpha: Tensor, color_change: Tensor, image: Tensor):
    image_rgb = image[:, 0:3, :, :]
    color_change_rgb = color_change[:, 0:3, :, :]
    output_rgb = color_change_rgb * alpha + image_rgb * (1 - alpha)
    return torch.cat([output_rgb, image[:, 3:4, :, :]], dim=1)


def apply_grid_change(grid_change, image: Tensor) -> Tensor:
    n, c, h, w = image.shape
    device = grid_change.device
    grid_change = torch.transpose(grid_change.view(n, 2, h * w), 1, 2).view(n, h, w, 2)
    identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device).unsqueeze(0).repeat(n, 1, 1)
    base_grid = affine_grid(identity, [n, c, h, w], align_corners=False)
    grid = base_grid + grid_change
    resampled_image = grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=False)
    return resampled_image


def apply_color_change(alpha, color_change, image: Tensor) -> Tensor:
    return color_change * alpha + image * (1 - alpha)
