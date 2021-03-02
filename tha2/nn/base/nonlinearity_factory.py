from typing import Optional

from torch.nn import Module, ReLU, LeakyReLU, ELU

from tha2.nn.base.module_factory import ModuleFactory


class ReLUFactory(ModuleFactory):
    def __init__(self, inplace: bool = False):
        self.inplace = inplace

    def create(self) -> Module:
        return ReLU(self.inplace)


class LeakyReLUFactory(ModuleFactory):
    def __init__(self, inplace: bool = False, negative_slope: float = 1e-2):
        self.negative_slope = negative_slope
        self.inplace = inplace

    def create(self) -> Module:
        return LeakyReLU(inplace=self.inplace, negative_slope=self.negative_slope)


class ELUFactory(ModuleFactory):
    def __init__(self, inplace: bool = False, alpha: float = 1.0):
        self.alpha = alpha
        self.inplace = inplace

    def create(self) -> Module:
        return ELU(inplace=self.inplace, alpha=self.alpha)


def resolve_nonlinearity_factory(nonlinearity_fatory: Optional[ModuleFactory]) -> ModuleFactory:
    if nonlinearity_fatory is None:
        return ReLUFactory(inplace=True)
    else:
        return nonlinearity_fatory
