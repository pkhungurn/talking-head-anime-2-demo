from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, List, Optional

from torch import Tensor


class PoseParameterCategory(Enum):
    EYEBROW = 1
    EYE = 2
    IRIS_MORPH = 3
    IRIS_ROTATION = 4
    MOUTH = 5
    FACE_ROTATION = 6


class PoseParameterGroup:
    def __init__(self,
                 group_name: str,
                 parameter_index: int,
                 category: PoseParameterCategory,
                 arity: int = 1,
                 discrete: bool = False,
                 default_value: float = 0.0,
                 range: Optional[Tuple[float, float]] = None):
        assert arity == 1 or arity == 2
        if range is None:
            range = (0.0, 1.0)
        if arity == 1:
            parameter_names = [group_name]
        else:
            parameter_names = [group_name + "_left", group_name + "_right"]
        assert len(parameter_names) == arity

        self.parameter_names = parameter_names
        self.range = range
        self.default_value = default_value
        self.discrete = discrete
        self.arity = arity
        self.category = category
        self.parameter_index = parameter_index
        self.group_name = group_name

    def get_arity(self) -> int:
        return self.arity

    def get_group_name(self) -> str:
        return self.group_name

    def get_parameter_names(self) -> List[str]:
        return self.parameter_names

    def is_discrete(self) -> bool:
        return self.discrete

    def get_range(self) -> Tuple[float, float]:
        return self.range

    def get_default_value(self):
        return self.default_value

    def get_parameter_index(self):
        return self.parameter_index

    def get_category(self) -> PoseParameterCategory:
        return self.category


class PoseParameters:
    def __init__(self, pose_parameter_groups: List[PoseParameterGroup]):
        self.pose_parameter_groups = pose_parameter_groups

    def get_parameter_index(self, name: str) -> int:
        index = 0
        for parameter_group in self.pose_parameter_groups:
            for param_name in parameter_group.parameter_names:
                if name == param_name:
                    return index
                index += 1
        raise RuntimeError("Cannot find parameter with name %s" % name)

    def get_parameter_name(self, index: int) -> str:
        assert index >= 0 and index < self.get_parameter_count()

        for group in self.pose_parameter_groups:
            if index < group.get_arity():
                return group.get_parameter_names()[index]
            index -= group.arity

        raise RuntimeError("Something is wrong here!!!")

    def get_pose_parameter_groups(self):
        return self.pose_parameter_groups

    def get_parameter_count(self):
        count = 0
        for group in self.pose_parameter_groups:
            count += group.arity
        return count

    class Builder:
        def __init__(self):
            self.index = 0
            self.pose_parameter_groups = []

        def add_parameter_group(self,
                                group_name: str,
                                category: PoseParameterCategory,
                                arity: int = 1,
                                discrete: bool = False,
                                default_value: float = 0.0,
                                range: Optional[Tuple[float, float]] = None):
            self.pose_parameter_groups.append(
                PoseParameterGroup(
                    group_name,
                    self.index,
                    category,
                    arity,
                    discrete,
                    default_value,
                    range))
            self.index += arity
            return self

        def build(self) -> 'PoseParameters':
            return PoseParameters(self.pose_parameter_groups)


class Poser(ABC):
    @abstractmethod
    def get_output_length(self) -> int:
        pass

    @abstractmethod
    def get_pose_parameter_groups(self) -> List[PoseParameterGroup]:
        pass

    @abstractmethod
    def get_num_parameters(self) -> int:
        pass

    @abstractmethod
    def pose(self, image: Tensor, pose: Tensor, output_index: int = 0) -> Tensor:
        pass

    @abstractmethod
    def get_posing_outputs(self, image: Tensor, pose: Tensor) -> List[Tensor]:
        pass
