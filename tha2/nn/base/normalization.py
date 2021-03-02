from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch.nn import Module, BatchNorm2d, InstanceNorm2d, Parameter
from torch.nn.init import normal_, constant_

from tha2.nn.base.pass_through import PassThrough


class PixelNormalization(Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x / torch.sqrt((x ** 2).mean(dim=1, keepdim=True) + self.epsilon)


class NormalizationLayerFactory(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def create(self, num_features: int, affine: bool = True) -> Module:
        pass

    @staticmethod
    def resolve_2d(factory: Optional['NormalizationLayerFactory']) -> 'NormalizationLayerFactory':
        if factory is None:
            return InstanceNorm2dFactory()
        else:
            return factory


class Bias2d(Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.bias = Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        return x + self.bias


class NoNorm2dFactory(NormalizationLayerFactory):
    def __init__(self):
        super().__init__()

    def create(self, num_features: int, affine: bool = True) -> Module:
        if affine:
            return Bias2d(num_features)
        else:
            return PassThrough()


class BatchNorm2dFactory(NormalizationLayerFactory):
    def __init__(self,
                 weight_mean: Optional[float] = None,
                 weight_std: Optional[float] = None,
                 bias: Optional[float] = None):
        super().__init__()
        self.bias = bias
        self.weight_std = weight_std
        self.weight_mean = weight_mean

    def get_weight_mean(self):
        if self.weight_mean is None:
            return 1.0
        else:
            return self.weight_mean

    def get_weight_std(self):
        if self.weight_std is None:
            return 0.02
        else:
            return self.weight_std

    def create(self, num_features: int, affine: bool = True) -> Module:
        module = BatchNorm2d(num_features=num_features, affine=affine)
        if affine:
            if self.weight_mean is not None or self.weight_std is not None:
                normal_(module.weight, self.get_weight_mean(), self.get_weight_std())
            if self.bias is not None:
                constant_(module.bias, self.bias)
        return module


class InstanceNorm2dFactory(NormalizationLayerFactory):
    def __init__(self):
        super().__init__()

    def create(self, num_features: int, affine: bool = True) -> Module:
        return InstanceNorm2d(num_features=num_features, affine=affine)


class PixelNormFactory(NormalizationLayerFactory):
    def __init__(self):
        super().__init__()

    def create(self, num_features: int, affine: bool = True) -> Module:
        return PixelNormalization()