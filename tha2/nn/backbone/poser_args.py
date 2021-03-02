from typing import Optional

from torch.nn import Sigmoid, Sequential, Tanh

from tha2.nn.base.conv import create_conv3, create_conv3_from_block_args
from tha2.nn.base.nonlinearity_factory import ReLUFactory
from tha2.nn.base.normalization import InstanceNorm2dFactory
from tha2.nn.base.util import BlockArgs


class PoserArgs00:
    def __init__(self,
                 image_size: int,
                 input_image_channels: int,
                 output_image_channels: int,
                 start_channels: int,
                 num_pose_params: int,
                 block_args: Optional[BlockArgs] = None):
        self.num_pose_params = num_pose_params
        self.start_channels = start_channels
        self.output_image_channels = output_image_channels
        self.input_image_channels = input_image_channels
        self.image_size = image_size
        if block_args is None:
            self.block_args = BlockArgs(
                normalization_layer_factory=InstanceNorm2dFactory(),
                nonlinearity_factory=ReLUFactory(inplace=True))
        else:
            self.block_args = block_args

    def create_alpha_block(self):
        from torch.nn import Sequential
        return Sequential(
            create_conv3(
                in_channels=self.start_channels,
                out_channels=1,
                bias=True,
                initialization_method=self.block_args.initialization_method,
                use_spectral_norm=False),
            Sigmoid())

    def create_all_channel_alpha_block(self):
        from torch.nn import Sequential
        return Sequential(
            create_conv3(
                in_channels=self.start_channels,
                out_channels=self.output_image_channels,
                bias=True,
                initialization_method=self.block_args.initialization_method,
                use_spectral_norm=False),
            Sigmoid())

    def create_color_change_block(self):
        return Sequential(
            create_conv3_from_block_args(
                in_channels=self.start_channels,
                out_channels=self.output_image_channels,
                bias=True,
                block_args=self.block_args),
            Tanh())

    def create_grid_change_block(self):
        return create_conv3(
            in_channels=self.start_channels,
            out_channels=2,
            bias=False,
            initialization_method='zero',
            use_spectral_norm=False)