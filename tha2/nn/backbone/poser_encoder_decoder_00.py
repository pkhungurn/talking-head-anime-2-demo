import math
from typing import Optional, List

import torch
from torch import Tensor
from torch.nn import ModuleList, Module

from tha2.nn.backbone.poser_args import PoserArgs00
from tha2.nn.base.conv import create_conv3_block_from_block_args, create_downsample_block_from_block_args, \
    create_upsample_block_from_block_args
from tha2.nn.base.nonlinearity_factory import ReLUFactory
from tha2.nn.base.normalization import InstanceNorm2dFactory
from tha2.nn.base.resnet_block import ResnetBlock
from tha2.nn.base.util import BlockArgs


class PoserEncoderDecoder00Args(PoserArgs00):
    def __init__(self,
                 image_size: int,
                 input_image_channels: int,
                 output_image_channels: int,
                 num_pose_params: int ,
                 start_channels: int,
                 bottleneck_image_size,
                 num_bottleneck_blocks,
                 max_channels: int,
                 block_args: Optional[BlockArgs] = None):
        super().__init__(
            image_size, input_image_channels, output_image_channels, start_channels, num_pose_params, block_args)
        self.max_channels = max_channels
        self.num_bottleneck_blocks = num_bottleneck_blocks
        self.bottleneck_image_size = bottleneck_image_size
        assert bottleneck_image_size > 1

        if block_args is None:
            self.block_args = BlockArgs(
                normalization_layer_factory=InstanceNorm2dFactory(),
                nonlinearity_factory=ReLUFactory(inplace=True))
        else:
            self.block_args = block_args


class PoserEncoderDecoder00(Module):
    def __init__(self, args: PoserEncoderDecoder00Args):
        super().__init__()
        self.args = args

        self.num_levels = int(math.log2(args.image_size // args.bottleneck_image_size)) + 1

        self.downsample_blocks = ModuleList()
        self.downsample_blocks.append(
            create_conv3_block_from_block_args(
                args.input_image_channels,
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
            in_channels=current_num_channels + args.num_pose_params,
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

    def get_num_output_channels_from_level(self, level: int):
        return self.get_num_output_channels_from_image_size(self.args.image_size // (2 ** level))

    def get_num_output_channels_from_image_size(self, image_size: int):
        return min(self.args.start_channels * (self.args.image_size // image_size), self.args.max_channels)

    def forward(self, image: Tensor, pose: Optional[Tensor] = None) -> List[Tensor]:
        if self.args.num_pose_params != 0:
            assert pose is not None
        else:
            assert pose is None
        outputs = []
        feature = image
        outputs.append(feature)
        for block in self.downsample_blocks:
            feature = block(feature)
            outputs.append(feature)
        if pose is not None:
            n, c = pose.shape
            pose = pose.view(n, c, 1, 1).repeat(1, 1, self.args.bottleneck_image_size, self.args.bottleneck_image_size)
            feature = torch.cat([feature, pose], dim=1)
        for block in self.bottleneck_blocks:
            feature = block(feature)
            outputs.append(feature)
        for block in self.upsample_blocks:
            feature = block(feature)
            outputs.append(feature)
        outputs.reverse()
        return outputs
