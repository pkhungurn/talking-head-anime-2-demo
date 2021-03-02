import math
from enum import Enum
from typing import List, Dict, Optional

import torch
from torch import Tensor

from tha2.compute.cached_computation_func import TensorListCachedComputationFunc
from tha2.compute.cached_computation_protocol import CachedComputationProtocol
from tha2.nn.backcomp.tha.combiner import CombinerFactory
from tha2.nn.backcomp.tha.two_algo_face_rotator import TwoAlgoFaceRotatorFactory, TwoAlgoFaceRotator
from tha2.nn.base.nonlinearity_factory import ReLUFactory
from tha2.nn.base.normalization import InstanceNorm2dFactory
from tha2.nn.base.util import BlockArgs
from tha2.nn.batch_module.batch_input_module import BatchInputModule
from tha2.nn.eyebrow.eyebrow_decomposer_00 import EyebrowDecomposer00, \
    EyebrowDecomposer00Factory, EyebrowDecomposer00Args
from tha2.nn.eyebrow.eyebrow_morphing_combiner_00 import \
    EyebrowMorphingCombiner00Factory, EyebrowMorphingCombiner00Args, EyebrowMorphingCombiner00
from tha2.nn.face.face_morpher_08 import FaceMorpher08Args, FaceMorpher08Factory
from tha2.poser.general_poser_02 import GeneralPoser02
from tha2.poser.poser import Poser, PoseParameterCategory, PoseParameters
from tha2.util import torch_load

KEY_EYEBROW_DECOMPOSER = "eyebrow_decomposer"
KEY_EYEBROW_MORPHING_COMBINER = "eyebrow_combiner"
KEY_FACE_MORPHER = "face_morpher"
KEY_FACE_ROTATER = "face_rotator"
KEY_COMBINER = "combiner"

KEY_EYEBROW_DECOMPOSER_OUTPUT = "eyebrow_decomposer_output"
KEY_EYEBROW_MORPHING_COMBINER_OUTPUT = "eyebrow_combiner_output"
KEY_FACE_MORPHER_OUTPUT = "face_morpher_output"
KEY_FACE_ROTATER_OUTPUT = "face_rotator_output"
KEY_COMBINER_OUTPUT = "combiner_output"
KEY_ALL_OUTPUT = "all_output"

NUM_EYEBROW_PARAMS = 12
NUM_FACE_PARAMS = 27
NUM_ROTATION_PARAMS = 3


class FiveStepPoserComputationProtocol(CachedComputationProtocol):
    def __init__(self, eyebrow_morphed_image_index: int):
        super().__init__()
        self.eyebrow_morphed_image_index = eyebrow_morphed_image_index
        self.cached_batch_0 = None
        self.cached_eyebrow_decomposer_output = None

    def compute_func(self) -> TensorListCachedComputationFunc:
        def func(modules: Dict[str, BatchInputModule],
                 batch: List[Tensor],
                 outputs: Dict[str, List[Tensor]]):
            new_batch_0 = self.cached_batch_0 is None or torch.max((batch[0] - self.cached_batch_0).abs()).item() > 0
            if not new_batch_0:
                outputs[KEY_EYEBROW_DECOMPOSER_OUTPUT] = self.cached_eyebrow_decomposer_output
            output = self.get_output(KEY_ALL_OUTPUT, modules, batch, outputs)
            if new_batch_0:
                self.cached_batch_0 = batch[0]
                self.cached_eyebrow_decomposer_output = outputs[KEY_EYEBROW_DECOMPOSER_OUTPUT]
            return output

        return func

    def compute_output(self, key: str, modules: Dict[str, BatchInputModule], batch: List[Tensor],
                       outputs: Dict[str, List[Tensor]]) -> List[Tensor]:
        if key == KEY_EYEBROW_DECOMPOSER_OUTPUT:
            input_image = batch[0][:, :, 64:192, 64:192]
            return modules[KEY_EYEBROW_DECOMPOSER].forward_from_batch([input_image])
        elif key == KEY_EYEBROW_MORPHING_COMBINER_OUTPUT:
            eyebrow_decomposer_output = self.get_output(KEY_EYEBROW_DECOMPOSER_OUTPUT, modules, batch, outputs)
            background_layer = eyebrow_decomposer_output[EyebrowDecomposer00.BACKGROUND_LAYER_INDEX]
            eyebrow_layer = eyebrow_decomposer_output[EyebrowDecomposer00.EYEBROW_LAYER_INDEX]
            eyebrow_pose = batch[1][:, :NUM_EYEBROW_PARAMS]
            return modules[KEY_EYEBROW_MORPHING_COMBINER].forward_from_batch([
                background_layer,
                eyebrow_layer,
                eyebrow_pose
            ])
        elif key == KEY_FACE_MORPHER_OUTPUT:
            eyebrow_morphing_combiner_output = self.get_output(
                KEY_EYEBROW_MORPHING_COMBINER_OUTPUT, modules, batch, outputs)
            eyebrow_morphed_image = eyebrow_morphing_combiner_output[self.eyebrow_morphed_image_index]
            input_image = batch[0][:, :, 32:32 + 192, 32:32 + 192].clone()
            input_image[:, :, 32:32 + 128, 32:32 + 128] = eyebrow_morphed_image
            face_pose = batch[1][:, NUM_EYEBROW_PARAMS:NUM_EYEBROW_PARAMS + NUM_FACE_PARAMS]
            return modules[KEY_FACE_MORPHER].forward_from_batch([
                input_image,
                face_pose
            ])
        elif key == KEY_FACE_ROTATER_OUTPUT:
            face_morpher_output = self.get_output(KEY_FACE_MORPHER_OUTPUT, modules, batch, outputs)
            face_morphed_image = face_morpher_output[0]
            input_image = batch[0].clone()
            input_image[:, :, 32:32 + 192, 32:32 + 192] = face_morphed_image
            rotation_pose = batch[1][:, NUM_EYEBROW_PARAMS + NUM_FACE_PARAMS:]
            return modules[KEY_FACE_ROTATER].forward_from_batch([
                input_image,
                rotation_pose
            ])
        elif key == KEY_COMBINER_OUTPUT:
            face_rotater_output = self.get_output(KEY_FACE_ROTATER_OUTPUT, modules, batch, outputs)
            color_changed_image = face_rotater_output[
                TwoAlgoFaceRotator.COLOR_CHANGED_IMAGE_INDEX]
            resampled_image = face_rotater_output[
                TwoAlgoFaceRotator.RESAMPLED_IMAGE_INDEX]
            rotation_pose = batch[1][:, NUM_EYEBROW_PARAMS + NUM_FACE_PARAMS:]
            return modules[KEY_COMBINER].forward_from_batch([
                color_changed_image,
                resampled_image,
                rotation_pose
            ])
        elif key == KEY_ALL_OUTPUT:
            combiner_output = self.get_output(KEY_COMBINER_OUTPUT, modules, batch, outputs)
            rotater_output = self.get_output(KEY_FACE_ROTATER_OUTPUT, modules, batch, outputs)
            face_morpher_output = self.get_output(KEY_FACE_MORPHER_OUTPUT, modules, batch, outputs)
            eyebrow_morphing_combiner_output = self.get_output(
                KEY_EYEBROW_MORPHING_COMBINER_OUTPUT, modules, batch, outputs)
            eyebrow_decomposer_output = self.get_output(KEY_EYEBROW_DECOMPOSER_OUTPUT, modules, batch, outputs)
            output = combiner_output \
                     + rotater_output \
                     + face_morpher_output \
                     + eyebrow_morphing_combiner_output \
                     + eyebrow_decomposer_output
            return output
        else:
            raise RuntimeError("Unsupported key: " + key)


def load_eyebrow_decomposer(file_name: str):
    factory = EyebrowDecomposer00Factory(
        EyebrowDecomposer00Args(
            image_size=128,
            image_channels=4,
            start_channels=64,
            bottleneck_image_size=16,
            num_bottleneck_blocks=6,
            max_channels=512,
            block_args=BlockArgs(
                initialization_method='he',
                use_spectral_norm=False,
                normalization_layer_factory=InstanceNorm2dFactory(),
                nonlinearity_factory=ReLUFactory(inplace=True))))
    #print("Loading the eyebrow decomposer ... ", end="")
    module = factory.create()
    module.load_state_dict(torch_load(file_name))
    #print("DONE!!!")
    return module


def load_eyebrow_morphing_combiner(file_name: str):
    factory = EyebrowMorphingCombiner00Factory(
        EyebrowMorphingCombiner00Args(
            image_size=128,
            image_channels=4,
            start_channels=64,
            num_pose_params=12,
            bottleneck_image_size=16,
            num_bottleneck_blocks=6,
            max_channels=512,
            block_args=BlockArgs(
                initialization_method='he',
                use_spectral_norm=False,
                normalization_layer_factory=InstanceNorm2dFactory(),
                nonlinearity_factory=ReLUFactory(inplace=True))))
    #print("Loading the eyebrow morphing conbiner ... ", end="")
    module = factory.create()
    module.load_state_dict(torch_load(file_name))
    #print("DONE!!!")
    return module


def load_face_morpher(file_name: str):
    factory = FaceMorpher08Factory(
        FaceMorpher08Args(
            image_size=192,
            image_channels=4,
            num_expression_params=27,
            start_channels=64,
            bottleneck_image_size=24,
            num_bottleneck_blocks=6,
            max_channels=512,
            block_args=BlockArgs(
                initialization_method='he',
                use_spectral_norm=False,
                normalization_layer_factory=InstanceNorm2dFactory(),
                nonlinearity_factory=ReLUFactory(inplace=False))))
    #print("Loading the face morpher ... ", end="")
    module = factory.create()
    module.load_state_dict(torch_load(file_name))
    #print("DONE")
    return module


def load_face_rotater(file_name: str):
    #print("Loading the face rotater ... ", end="")
    module = TwoAlgoFaceRotatorFactory().create()
    module.load_state_dict(torch_load(file_name))
    #print("DONE!!!")
    return module


def load_combiner(file_name: str):
    #print("Loading the combiner ... ", end="")
    module = CombinerFactory().create()
    module.load_state_dict(torch_load(file_name))
    #print("DONE!!!")
    return module


def get_pose_parameters():
    return PoseParameters.Builder() \
        .add_parameter_group("eyebrow_troubled", PoseParameterCategory.EYEBROW, arity=2) \
        .add_parameter_group("eyebrow_angry", PoseParameterCategory.EYEBROW, arity=2) \
        .add_parameter_group("eyebrow_lowered", PoseParameterCategory.EYEBROW, arity=2) \
        .add_parameter_group("eyebrow_raised", PoseParameterCategory.EYEBROW, arity=2) \
        .add_parameter_group("eyebrow_happy", PoseParameterCategory.EYEBROW, arity=2) \
        .add_parameter_group("eyebrow_serious", PoseParameterCategory.EYEBROW, arity=2) \
        .add_parameter_group("eye_wink", PoseParameterCategory.EYE, arity=2) \
        .add_parameter_group("eye_happy_wink", PoseParameterCategory.EYE, arity=2) \
        .add_parameter_group("eye_surprised", PoseParameterCategory.EYE, arity=2) \
        .add_parameter_group("eye_relaxed", PoseParameterCategory.EYE, arity=2) \
        .add_parameter_group("eye_unimpressed", PoseParameterCategory.EYE, arity=2) \
        .add_parameter_group("eye_raised_lower_eyelid", PoseParameterCategory.EYE, arity=2) \
        .add_parameter_group("iris_small", PoseParameterCategory.IRIS_MORPH, arity=2) \
        .add_parameter_group("mouth_aaa", PoseParameterCategory.MOUTH, arity=1, default_value=1.0) \
        .add_parameter_group("mouth_iii", PoseParameterCategory.MOUTH, arity=1) \
        .add_parameter_group("mouth_uuu", PoseParameterCategory.MOUTH, arity=1) \
        .add_parameter_group("mouth_eee", PoseParameterCategory.MOUTH, arity=1) \
        .add_parameter_group("mouth_ooo", PoseParameterCategory.MOUTH, arity=1) \
        .add_parameter_group("mouth_delta", PoseParameterCategory.MOUTH, arity=1) \
        .add_parameter_group("mouth_lowered_corner", PoseParameterCategory.MOUTH, arity=2) \
        .add_parameter_group("mouth_raised_corner", PoseParameterCategory.MOUTH, arity=2) \
        .add_parameter_group("mouth_smirk", PoseParameterCategory.MOUTH, arity=1) \
        .add_parameter_group("iris_rotation_x", PoseParameterCategory.IRIS_ROTATION, arity=1, range=(-1.0, 1.0)) \
        .add_parameter_group("iris_rotation_y", PoseParameterCategory.IRIS_ROTATION, arity=1, range=(-1.0, 1.0)) \
        .add_parameter_group("head_x", PoseParameterCategory.FACE_ROTATION, arity=1, range=(-1.0, 1.0)) \
        .add_parameter_group("head_y", PoseParameterCategory.FACE_ROTATION, arity=1, range=(-1.0, 1.0)) \
        .add_parameter_group("neck_z", PoseParameterCategory.FACE_ROTATION, arity=1, range=(-1.0, 1.0)) \
        .build()


def create_poser(
        device: torch.device,
        module_file_names: Optional[Dict[str, str]] = None,
        eyebrow_morphed_image_index: int = EyebrowMorphingCombiner00.EYEBROW_IMAGE_NO_COMBINE_ALPHA_INDEX,
        default_output_index: int = 0) -> Poser:
    dir = "data"
    if module_file_names is None:
        module_file_names = {}
    if KEY_EYEBROW_DECOMPOSER not in module_file_names:
        file_name = dir + "/eyebrow_decomposer.pt"
        module_file_names[KEY_EYEBROW_DECOMPOSER] = file_name
    if KEY_EYEBROW_MORPHING_COMBINER not in module_file_names:
        file_name = dir + "/eyebrow_morphing_combiner.pt"
        module_file_names[KEY_EYEBROW_MORPHING_COMBINER] = file_name
    if KEY_FACE_MORPHER not in module_file_names:
        file_name = dir + "/face_morpher.pt"
        module_file_names[KEY_FACE_MORPHER] = file_name
    if KEY_FACE_ROTATER not in module_file_names:
        file_name = dir + "/two_algo_face_rotator.pt"
        module_file_names[KEY_FACE_ROTATER] = file_name
    if KEY_COMBINER not in module_file_names:
        file_name = dir + "/combiner.pt"
        module_file_names[KEY_COMBINER] = file_name

    loaders = {
        KEY_EYEBROW_DECOMPOSER:
            lambda: load_eyebrow_decomposer(module_file_names[KEY_EYEBROW_DECOMPOSER]),
        KEY_EYEBROW_MORPHING_COMBINER:
            lambda: load_eyebrow_morphing_combiner(module_file_names[KEY_EYEBROW_MORPHING_COMBINER]),
        KEY_FACE_MORPHER:
            lambda: load_face_morpher(module_file_names[KEY_FACE_MORPHER]),
        KEY_FACE_ROTATER:
            lambda: load_face_rotater(module_file_names[KEY_FACE_ROTATER]),
        KEY_COMBINER:
            lambda: load_combiner(module_file_names[KEY_COMBINER]),
    }
    return GeneralPoser02(
        module_loaders=loaders,
        pose_parameters=get_pose_parameters().get_pose_parameter_groups(),
        output_list_func=FiveStepPoserComputationProtocol(eyebrow_morphed_image_index).compute_func(),
        subrect=None,
        device=device,
        output_length=31,
        default_output_index=default_output_index)