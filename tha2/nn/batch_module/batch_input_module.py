from abc import ABC, abstractmethod
from typing import List

from torch import Tensor
from torch.nn import Module

from tha2.nn.base.module_factory import ModuleFactory


class BatchInputModule(Module, ABC):
    def __init__(self, num_run_args=0):
        super().__init__()

        self.num_run_args = num_run_args

    def forward_from_batch(self, batch: List[Tensor]):
        assert len(batch) >= self.num_run_args, f"At least {self.num_run_args} arg(s) should be provided."
        if len(batch) > self.num_run_args:
            batch = batch[:self.num_run_args]
        return self.forward(*batch)


class BatchInputModuleFactory(ModuleFactory):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def create(self) -> BatchInputModule:
        pass
