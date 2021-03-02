from typing import List, Optional, Tuple, Dict, Callable

import torch
from torch import Tensor

from tha2.poser.poser import PoseParameterGroup, Poser
from tha2.nn.batch_module.batch_input_module import BatchInputModule
from tha2.compute.cached_computation_func import TensorListCachedComputationFunc


class GeneralPoser02(Poser):
    def __init__(self,
                 module_loaders: Dict[str, Callable[[], BatchInputModule]],
                 device: torch.device,
                 output_length: int,
                 pose_parameters: List[PoseParameterGroup],
                 output_list_func: TensorListCachedComputationFunc,
                 subrect: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
                 default_output_index: int = 0):
        self.default_output_index = default_output_index
        self.output_list_func = output_list_func
        self.subrect = subrect
        self.pose_parameters = pose_parameters
        self.device = device
        self.module_loaders = module_loaders

        self.modules = None

        self.num_parameters = 0
        for pose_parameter in self.pose_parameters:
            self.num_parameters += pose_parameter.get_arity()

        self.output_length = output_length

    def get_modules(self):
        if self.modules is None:
            self.modules = {}
            for key in self.module_loaders:
                module = self.module_loaders[key]()
                self.modules[key] = module
                module.to(self.device)
                module.train(False)
        return self.modules

    def get_pose_parameter_groups(self) -> List[PoseParameterGroup]:
        return self.pose_parameters

    def get_num_parameters(self) -> int:
        return self.num_parameters

    def pose(self, image: Tensor, pose: Tensor, output_index: Optional[int] = None) -> Tensor:
        if output_index is None:
            output_index = self.default_output_index
        output_list = self.get_posing_outputs(image, pose)
        return output_list[output_index]

    def get_posing_outputs(self, image: Tensor, pose: Tensor) -> List[Tensor]:
        modules = self.get_modules()

        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        if len(pose.shape) == 1:
            pose = pose.unsqueeze(0)
        if self.subrect is not None:
            image = image[:, :, self.subrect[0][0]:self.subrect[0][1], self.subrect[1][0]:self.subrect[1][1]]
        batch = [image, pose]

        outputs = {}
        return self.output_list_func(modules, batch, outputs)

    def get_output_length(self) -> int:
        return self.output_length
