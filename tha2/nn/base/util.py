from typing import Optional

from torch.nn import Module

from tha2.nn.base.init_function import create_init_function
from tha2.nn.base.module_factory import ModuleFactory
from tha2.nn.base.nonlinearity_factory import resolve_nonlinearity_factory
from tha2.nn.base.normalization import NormalizationLayerFactory
from tha2.nn.base.spectral_norm import apply_spectral_norm


def wrap_conv_or_linear_module(module: Module, initialization_method: str, use_spectral_norm: bool):
    init = create_init_function(initialization_method)
    return apply_spectral_norm(init(module), use_spectral_norm)


class ImageArgs:
    def __init__(self, size: int = 64, num_channels: int = 3):
        self.num_channels = num_channels
        self.size = size


class BlockArgs:
    def __init__(self,
                 initialization_method: str = 'he',
                 use_spectral_norm: bool = False,
                 normalization_layer_factory: Optional[NormalizationLayerFactory] = None,
                 nonlinearity_factory: Optional[ModuleFactory] = None):
        self.nonlinearity_factory = resolve_nonlinearity_factory(nonlinearity_factory)
        self.normalization_layer_factory = normalization_layer_factory
        self.use_spectral_norm = use_spectral_norm
        self.initialization_method = initialization_method

    def wrap_module(self, module: Module) -> Module:
        return wrap_conv_or_linear_module(module, self.initialization_method, self.use_spectral_norm)
