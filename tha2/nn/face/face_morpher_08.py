import math
from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import ModuleList, Sequential, Sigmoid, Tanh
from torch.nn.functional import affine_grid, grid_sample

from tha2.nn.batch_module.batch_input_module import BatchInputModule, BatchInputModuleFactory
from tha2.nn.base.conv import create_conv3_block_from_block_args, \
    create_downsample_block_from_block_args, create_upsample_block_from_block_args, create_conv3_from_block_args, \
    create_conv3
from tha2.nn.base.nonlinearity_factory import LeakyReLUFactory
from tha2.nn.base.normalization import InstanceNorm2dFactory
from tha2.nn.base.resnet_block import ResnetBlock
from tha2.nn.base.util import BlockArgs


class FaceMorpher08Args:
    def __init__(self,
                 image_size: int = 256,
                 image_channels: int = 4,
                 num_expression_params: int = 67,
                 start_channels: int = 16,
                 bottleneck_image_size=4,
                 num_bottleneck_blocks=3,
                 max_channels: int = 512,
                 block_args: Optional[BlockArgs] = None):
        self.max_channels = max_channels
        self.num_bottleneck_blocks = num_bottleneck_blocks
        assert bottleneck_image_size > 1
        self.bottleneck_image_size = bottleneck_image_size
        self.start_channels = start_channels
        self.image_channels = image_channels
        self.num_expression_params = num_expression_params
        self.image_size = image_size

        if block_args is None:
            self.block_args = BlockArgs(
                normalization_layer_factory=InstanceNorm2dFactory(),
                nonlinearity_factory=LeakyReLUFactory(negative_slope=0.2, inplace=True))
        else:
            self.block_args = block_args


class FaceMorpher08(BatchInputModule):
    def __init__(self, args: FaceMorpher08Args):
        super().__init__()
        self.args = args
        self.num_levels = int(math.log2(args.image_size // args.bottleneck_image_size)) + 1

        self.downsample_blocks = ModuleList()
        self.downsample_blocks.append(
            create_conv3_block_from_block_args(
                args.image_channels,
                args.start_channels,
                args.block_args))
        current_image_size = args.image_size
        current_num_channels = args.start_channels
        while current_image_size > args.bottleneck_image_size:
            next_image_size = current_image_size // 2
            next_num_channels = self.get_num_output_channels_from_image_size(next_image_size)
            self.downsample_blocks.append(create_downsample_block_from_block_args(
                in_channels=current_num_channels,
                out_channels=next_num_channels,
                is_output_1x1=False,
                block_args=args.block_args))
            current_image_size = next_image_size
            current_num_channels = next_num_channels
        assert len(self.downsample_blocks) == self.num_levels

        self.bottleneck_blocks = ModuleList()
        self.bottleneck_blocks.append(create_conv3_block_from_block_args(
            in_channels=current_num_channels + args.num_expression_params,
            out_channels=current_num_channels,
            block_args=args.block_args))
        for i in range(1, args.num_bottleneck_blocks):
            self.bottleneck_blocks.append(
                ResnetBlock.create(
                    num_channels=current_num_channels,
                    is1x1=False,
                    block_args=args.block_args))

        self.upsample_blocks = ModuleList()
        while current_image_size < args.image_size:
            next_image_size = current_image_size * 2
            next_num_channels = self.get_num_output_channels_from_image_size(next_image_size)
            self.upsample_blocks.append(create_upsample_block_from_block_args(
                in_channels=current_num_channels,
                out_channels=next_num_channels,
                block_args=args.block_args))
            current_image_size = next_image_size
            current_num_channels = next_num_channels

        self.iris_mouth_grid_change = self.create_grid_change_block()
        self.iris_mouth_color_change = self.create_color_change_block()
        self.iris_mouth_alpha = self.create_alpha_block()

        self.eye_color_change = self.create_color_change_block()
        self.eye_alpha = self.create_alpha_block()

    def create_alpha_block(self):
        return Sequential(
            create_conv3(
                in_channels=self.args.start_channels,
                out_channels=1,
                bias=True,
                initialization_method=self.args.block_args.initialization_method,
                use_spectral_norm=False),
            Sigmoid())

    def create_color_change_block(self):
        return Sequential(
            create_conv3_from_block_args(
                in_channels=self.args.start_channels,
                out_channels=self.args.image_channels,
                bias=True,
                block_args=self.args.block_args),
            Tanh())

    def create_grid_change_block(self):
        return create_conv3(
            in_channels=self.args.start_channels,
            out_channels=2,
            bias=False,
            initialization_method='zero',
            use_spectral_norm=False)

    def get_num_output_channels_from_level(self, level: int):
        return self.get_num_output_channels_from_image_size(self.args.image_size // (2 ** level))

    def get_num_output_channels_from_image_size(self, image_size: int):
        return min(self.args.start_channels * (self.args.image_size // image_size), self.args.max_channels)

    def forward(self, image: Tensor, pose: Tensor) -> List[Tensor]:
        feature = image
        for block in self.downsample_blocks:
            feature = block(feature)
        n, c = pose.shape
        pose = pose.view(n, c, 1, 1).repeat(1, 1, self.args.bottleneck_image_size, self.args.bottleneck_image_size)
        feature = torch.cat([feature, pose], dim=1)
        for block in self.bottleneck_blocks:
            feature = block(feature)
        for block in self.upsample_blocks:
            feature = block(feature)

        iris_mouth_grid_change = self.iris_mouth_grid_change(feature)
        iris_mouth_image_0 = self.apply_grid_change(iris_mouth_grid_change, image)
        iris_mouth_color_change = self.iris_mouth_color_change(feature)
        iris_mouth_alpha = self.iris_mouth_alpha(feature)
        iris_mouth_image_1 = self.apply_color_change(iris_mouth_alpha, iris_mouth_color_change, iris_mouth_image_0)

        eye_color_change = self.eye_color_change(feature)
        eye_alpha = self.eye_alpha(feature)
        output_image = self.apply_color_change(eye_alpha, eye_color_change, iris_mouth_image_1.detach())

        return [
            output_image, #0
            eye_alpha, #1
            eye_color_change, #2
            iris_mouth_image_1, #3
            iris_mouth_alpha, #4
            iris_mouth_color_change, #5
            iris_mouth_image_0, #6
        ]

    def merge_down(self, top_layer: Tensor, bottom_layer: Tensor):
        top_layer_rgb = top_layer[:, 0:3, :, :]
        top_layer_a = top_layer[:, 3:4, :, :]
        return bottom_layer * (1-top_layer_a) + torch.cat([top_layer_rgb * top_layer_a, top_layer_a], dim=1)

    def apply_grid_change(self, grid_change, image: Tensor) -> Tensor:
        n, c, h, w = image.shape
        device = grid_change.device
        grid_change = torch.transpose(grid_change.view(n, 2, h * w), 1, 2).view(n, h, w, 2)
        identity = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device).unsqueeze(0).repeat(n, 1, 1)
        base_grid = affine_grid(identity, [n, c, h, w], align_corners=False)
        grid = base_grid + grid_change
        resampled_image = grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=False)
        return resampled_image

    def apply_color_change(self, alpha, color_change, image: Tensor) -> Tensor:
        return color_change * alpha + image * (1 - alpha)

    def forward_from_batch(self, batch: List[Tensor]):
        return self.forward(batch[0], batch[1])


class FaceMorpher08Factory(BatchInputModuleFactory):
    def __init__(self, args: FaceMorpher08Args):
        super().__init__()
        self.args = args

    def create(self) -> BatchInputModule:
        return FaceMorpher08(self.args)


if __name__ == "__main__":
    cuda = torch.device('cuda')
    args = FaceMorpher08Args(
        image_size=256,
        image_channels=4,
        num_expression_params=67,
        start_channels=32,
        bottleneck_image_size=4,
        num_bottleneck_blocks=3,
        block_args=BlockArgs(
            initialization_method='xavier',
            use_spectral_norm=False,
            normalization_layer_factory=InstanceNorm2dFactory(),
            nonlinearity_factory=LeakyReLUFactory(inplace=True, negative_slope=0.2)))
    face_morpher = FaceMorpher08(args).to(cuda)

    image = torch.randn(8, 4, 256, 256, device=cuda)
    pose = torch.randn(8, 67, device=cuda)
    outputs = face_morpher.forward(image, pose)
    for i in range(len(outputs)):
        print(i, outputs[i].shape)
