from typing import Optional

import torch
from torch.nn import Module, Sequential, Parameter

from tha2.nn.base.conv import create_conv1, create_conv3
from tha2.nn.base.module_factory import ModuleFactory
from tha2.nn.base.nonlinearity_factory import resolve_nonlinearity_factory
from tha2.nn.base.normalization import NormalizationLayerFactory
from tha2.nn.base.util import BlockArgs


class ResnetBlock(Module):
    @staticmethod
    def create(num_channels: int,
               is1x1: bool = False,
               use_scale_parameters: bool = False,
               block_args: Optional[BlockArgs] = None):
        if block_args is None:
            block_args = BlockArgs()
        return ResnetBlock(num_channels,
                           is1x1,
                           block_args.initialization_method,
                           block_args.nonlinearity_factory,
                           block_args.normalization_layer_factory,
                           block_args.use_spectral_norm,
                           use_scale_parameters)

    def __init__(self,
                 num_channels: int,
                 is1x1: bool = False,
                 initialization_method: str = 'he',
                 nonlinearity_factory: ModuleFactory = None,
                 normalization_layer_factory: Optional[NormalizationLayerFactory] = None,
                 use_spectral_norm: bool = False,
                 use_scale_parameter: bool = False):
        super().__init__()
        self.use_scale_parameter = use_scale_parameter
        if self.use_scale_parameter:
            self.scale = Parameter(torch.zeros(1))
        nonlinearity_factory = resolve_nonlinearity_factory(nonlinearity_factory)
        if is1x1:
            self.resnet_path = Sequential(
                create_conv1(num_channels, num_channels, initialization_method,
                             bias=True,
                             use_spectral_norm=use_spectral_norm),
                nonlinearity_factory.create(),
                create_conv1(num_channels, num_channels, initialization_method,
                             bias=True,
                             use_spectral_norm=use_spectral_norm))
        else:
            self.resnet_path = Sequential(
                create_conv3(num_channels, num_channels,
                             bias=False, initialization_method=initialization_method,
                             use_spectral_norm=use_spectral_norm),
                NormalizationLayerFactory.resolve_2d(normalization_layer_factory).create(num_channels, affine=True),
                nonlinearity_factory.create(),
                create_conv3(num_channels, num_channels,
                             bias=False, initialization_method=initialization_method,
                             use_spectral_norm=use_spectral_norm),
                NormalizationLayerFactory.resolve_2d(normalization_layer_factory).create(num_channels, affine=True))

    def forward(self, x):
        if self.use_scale_parameter:
            return x + self.scale * self.resnet_path(x)
        else:
            return x + self.resnet_path(x)
