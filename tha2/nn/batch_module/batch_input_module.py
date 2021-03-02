from abc import ABC, abstractmethod
from typing import List

from torch import Tensor
from torch.nn import Module

from tha2.nn.base.module_factory import ModuleFactory


class BatchInputModule(Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward_from_batch(self, batch: List[Tensor]):
        pass


class BatchInputModuleFactory(ModuleFactory):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def create(self) -> BatchInputModule:
        pass
