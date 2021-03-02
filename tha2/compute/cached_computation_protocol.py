from abc import ABC, abstractmethod
from typing import Dict, List

from torch import Tensor

from tha2.nn.batch_module.batch_input_module import BatchInputModule
from tha2.compute.cached_computation_func import TensorCachedComputationFunc, TensorListCachedComputationFunc


class CachedComputationProtocol(ABC):
    def get_output(self,
                   key: str,
                   modules: Dict[str, BatchInputModule],
                   batch: List[Tensor],
                   outputs: Dict[str, List[Tensor]]):
        if key in outputs:
            return outputs[key]
        else:
            output = self.compute_output(key, modules, batch, outputs)
            outputs[key] = output
            return outputs[key]

    @abstractmethod
    def compute_output(self,
                       key: str,
                       modules: Dict[str, BatchInputModule],
                       batch: List[Tensor],
                       outputs: Dict[str, List[Tensor]]) -> List[Tensor]:
        pass

    def get_output_tensor_func(self, key: str, index: int) -> TensorCachedComputationFunc:
        def func(modules: Dict[str, BatchInputModule],
                 batch: List[Tensor],
                 outputs: Dict[str, List[Tensor]]):
            return self.get_output(key, modules, batch, outputs)[index]
        return func

    def get_output_tensor_list_func(self, key: str) -> TensorListCachedComputationFunc:
        def func(modules: Dict[str, BatchInputModule],
                 batch: List[Tensor],
                 outputs: Dict[str, List[Tensor]]):
            return self.get_output(key, modules, batch, outputs)
        return func