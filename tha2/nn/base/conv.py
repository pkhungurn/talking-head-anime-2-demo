from typing import Optional

from torch.nn import Conv2d, Module, Sequential, ConvTranspose2d

from tha2.nn.base.module_factory import ModuleFactory
from tha2.nn.base.nonlinearity_factory import resolve_nonlinearity_factory
from tha2.nn.base.normalization import NormalizationLayerFactory
from tha2.nn.base.util import wrap_conv_or_linear_module, BlockArgs


def create_conv7(in_channels: int, out_channels: int,
                 bias: bool = False,
                 initialization_method='he',
                 use_spectral_norm: bool = False) -> Module:
    return wrap_conv_or_linear_module(
        Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=bias),
        initialization_method,
        use_spectral_norm)

def create_conv7_from_block_args(in_channels: int, out_channels: int,
                                 bias: bool = False,
                                 block_args: Optional[BlockArgs] = None)-> Module:
    if block_args is None:
        block_args = BlockArgs()
    return create_conv7(in_channels, out_channels, bias,
                        block_args.initialization_method,
                        block_args.use_spectral_norm)


def create_conv3(in_channels: int, out_channels: int,
                 bias: bool = False,
                 initialization_method='he',
                 use_spectral_norm: bool = False) -> Module:
    return wrap_conv_or_linear_module(
        Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
        initialization_method,
        use_spectral_norm)


def create_conv3_from_block_args(in_channels: int, out_channels: int,
                                 bias: bool = False,
                                 block_args: Optional[BlockArgs] = None):
    if block_args is None:
        block_args = BlockArgs()
    return create_conv3(in_channels, out_channels, bias,
                        block_args.initialization_method,
                        block_args.use_spectral_norm)


def create_conv1(in_channels: int, out_channels: int,
                 initialization_method='he',
                 bias: bool = False,
                 use_spectral_norm: bool = False) -> Module:
    return wrap_conv_or_linear_module(
        Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        initialization_method,
        use_spectral_norm)


def create_conv7_block(in_channels: int, out_channels: int,
                       initialization_method='he',
                       nonlinearity_factory: Optional[ModuleFactory] = None,
                       normalization_layer_factory: Optional[NormalizationLayerFactory] = None,
                       use_spectral_norm: bool = False) -> Module:
    nonlinearity_factory = resolve_nonlinearity_factory(nonlinearity_factory)
    return Sequential(
        create_conv7(in_channels, out_channels,
                     bias=False, initialization_method=initialization_method, use_spectral_norm=use_spectral_norm),
        NormalizationLayerFactory.resolve_2d(normalization_layer_factory).create(out_channels, affine=True),
        resolve_nonlinearity_factory(nonlinearity_factory).create())


def create_conv7_block_from_block_args(
        in_channels: int, out_channels: int,
        block_args: Optional[BlockArgs] = None) -> Module:
    if block_args is None:
        block_args = BlockArgs()
    return create_conv7_block(in_channels, out_channels,
                              block_args.initialization_method,
                              block_args.nonlinearity_factory,
                              block_args.normalization_layer_factory,
                              block_args.use_spectral_norm)


def create_conv3_block(in_channels: int, out_channels: int,
                       initialization_method='he',
                       nonlinearity_factory: Optional[ModuleFactory] = None,
                       normalization_layer_factory: Optional[NormalizationLayerFactory] = None,
                       use_spectral_norm: bool = False) -> Module:
    nonlinearity_factory = resolve_nonlinearity_factory(nonlinearity_factory)
    return Sequential(
        create_conv3(in_channels, out_channels,
                     bias=False, initialization_method=initialization_method, use_spectral_norm=use_spectral_norm),
        NormalizationLayerFactory.resolve_2d(normalization_layer_factory).create(out_channels, affine=True),
        resolve_nonlinearity_factory(nonlinearity_factory).create())


def create_conv3_block_from_block_args(
        in_channels: int, out_channels: int, block_args: Optional[BlockArgs] = None):
    if block_args is None:
        block_args = BlockArgs()
    return create_conv3_block(in_channels, out_channels,
                              block_args.initialization_method,
                              block_args.nonlinearity_factory,
                              block_args.normalization_layer_factory,
                              block_args.use_spectral_norm)


def create_downsample_block(in_channels: int, out_channels: int,
                            is_output_1x1: bool = False,
                            initialization_method='he',
                            nonlinearity_factory: Optional[ModuleFactory] = None,
                            normalization_layer_factory: Optional[NormalizationLayerFactory] = None,
                            use_spectral_norm: bool = False) -> Module:
    if is_output_1x1:
        return Sequential(
            wrap_conv_or_linear_module(
                Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                initialization_method,
                use_spectral_norm),
            resolve_nonlinearity_factory(nonlinearity_factory).create())
    else:
        return Sequential(
            wrap_conv_or_linear_module(
                Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                initialization_method,
                use_spectral_norm),
            NormalizationLayerFactory.resolve_2d(normalization_layer_factory).create(out_channels, affine=True),
            resolve_nonlinearity_factory(nonlinearity_factory).create())


def create_downsample_block_from_block_args(in_channels: int, out_channels: int,
                                            is_output_1x1: bool = False,
                                            block_args: Optional[BlockArgs] = None):
    if block_args is None:
        block_args = BlockArgs()
    return create_downsample_block(
        in_channels, out_channels,
        is_output_1x1,
        block_args.initialization_method,
        block_args.nonlinearity_factory,
        block_args.normalization_layer_factory,
        block_args.use_spectral_norm)


def create_upsample_block(in_channels: int,
                          out_channels: int,
                          initialization_method='he',
                          nonlinearity_factory: Optional[ModuleFactory] = None,
                          normalization_layer_factory: Optional[NormalizationLayerFactory] = None,
                          use_spectral_norm: bool = False) -> Module:
    nonlinearity_factory = resolve_nonlinearity_factory(nonlinearity_factory)
    return Sequential(
        wrap_conv_or_linear_module(
            ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            initialization_method,
            use_spectral_norm),
        NormalizationLayerFactory.resolve_2d(normalization_layer_factory).create(out_channels, affine=True),
        resolve_nonlinearity_factory(nonlinearity_factory).create())


def create_upsample_block_from_block_args(in_channels: int,
                                          out_channels: int,
                                          block_args: Optional[BlockArgs] = None) -> Module:
    if block_args is None:
        block_args = BlockArgs()
    return create_upsample_block(in_channels, out_channels,
                                 block_args.initialization_method,
                                 block_args.nonlinearity_factory,
                                 block_args.normalization_layer_factory,
                                 block_args.use_spectral_norm)
