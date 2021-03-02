from typing import Callable, Dict, List

from torch import Tensor

from tha2.nn.batch_module.batch_input_module import BatchInputModule

TensorCachedComputationFunc = Callable[
    [Dict[str, BatchInputModule], List[Tensor], Dict[str, List[Tensor]]], Tensor]
TensorListCachedComputationFunc = Callable[
    [Dict[str, BatchInputModule], List[Tensor], Dict[str, List[Tensor]]], List[Tensor]]
