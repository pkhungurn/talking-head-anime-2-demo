from abc import ABC, abstractmethod

from torch.nn import Module


class ModuleFactory(ABC):
    @abstractmethod
    def create(self) -> Module:
        pass