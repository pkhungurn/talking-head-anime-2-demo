from typing import List, Optional

import torch
from torch import Tensor

from tha2.nn.backbone.poser_encoder_decoder_00 import PoserEncoderDecoder00Args, PoserEncoderDecoder00
from tha2.nn.util import apply_color_change, apply_grid_change, apply_rgb_change
from tha2.nn.batch_module.batch_input_module import BatchInputModule, BatchInputModuleFactory
from tha2.nn.base.nonlinearity_factory import ReLUFactory
from tha2.nn.base.normalization import InstanceNorm2dFactory
from tha2.nn.base.util import BlockArgs


class EyebrowMorphingCombiner00Args(PoserEncoderDecoder00Args):
    def __init__(self,
                 image_size: int = 128,
                 image_channels: int = 4,
                 num_pose_params: int = 12,
                 start_channels: int = 64,
                 bottleneck_image_size=16,
                 num_bottleneck_blocks=6,
                 max_channels: int = 512,
                 block_args: Optional[BlockArgs] = None):
        super().__init__(
            image_size,
            2 * image_channels,
            image_channels,
            num_pose_params,
            start_channels,
            bottleneck_image_size,
            num_bottleneck_blocks,
            max_channels,
            block_args)


class EyebrowMorphingCombiner00(BatchInputModule):
    def __init__(self, args: EyebrowMorphingCombiner00Args):
        super().__init__()
        self.args = args
        self.body = PoserEncoderDecoder00(args)
        self.morphed_eyebrow_layer_grid_change = self.args.create_grid_change_block()
        self.morphed_eyebrow_layer_alpha = self.args.create_alpha_block()
        self.morphed_eyebrow_layer_color_change = self.args.create_color_change_block()
        self.combine_alpha = self.args.create_alpha_block()

    def forward(self, background_layer: Tensor, eyebrow_layer: Tensor, pose: Tensor) -> List[Tensor]:
        combined_image = torch.cat([background_layer, eyebrow_layer], dim=1)
        feature = self.body(combined_image, pose)[0]

        morphed_eyebrow_layer_grid_change = self.morphed_eyebrow_layer_grid_change(feature)
        morphed_eyebrow_layer_alpha = self.morphed_eyebrow_layer_alpha(feature)
        morphed_eyebrow_layer_color_change = self.morphed_eyebrow_layer_color_change(feature)
        warped_eyebrow_layer = apply_grid_change(morphed_eyebrow_layer_grid_change, eyebrow_layer)
        morphed_eyebrow_layer = apply_color_change(
            morphed_eyebrow_layer_alpha, morphed_eyebrow_layer_color_change, warped_eyebrow_layer)

        combine_alpha = self.combine_alpha(feature)
        eyebrow_image = apply_rgb_change(combine_alpha, morphed_eyebrow_layer, background_layer)
        eyebrow_image_no_combine_alpha = apply_rgb_change(
            (morphed_eyebrow_layer[:, 3:4, :, :] + 1.0) / 2.0, morphed_eyebrow_layer, background_layer)

        return [
            eyebrow_image,  # 0
            combine_alpha,  # 1
            eyebrow_image_no_combine_alpha,  # 2
            morphed_eyebrow_layer,  # 3
            morphed_eyebrow_layer_alpha,  # 4
            morphed_eyebrow_layer_color_change,  # 5
            warped_eyebrow_layer,  # 6
            morphed_eyebrow_layer_grid_change,  # 7
        ]

    EYEBROW_IMAGE_INDEX = 0
    COMBINE_ALPHA_INDEX = 1
    EYEBROW_IMAGE_NO_COMBINE_ALPHA_INDEX = 2
    MORPHED_EYEBROW_LAYER_INDEX = 3
    MORPHED_EYEBROW_LAYER_ALPHA_INDEX = 4
    MORPHED_EYEBROW_LAYER_COLOR_CHANGE_INDEX = 5
    WARPED_EYEBROW_LAYER_INDEX = 6
    MORPHED_EYEBROW_LAYER_GRID_CHANGE_INDEX = 7
    OUTPUT_LENGTH = 8

    def forward_from_batch(self, batch: List[Tensor]):
        return self.forward(batch[0], batch[1], batch[2])


class EyebrowMorphingCombiner00Factory(BatchInputModuleFactory):
    def __init__(self, args: EyebrowMorphingCombiner00Args):
        super().__init__()
        self.args = args

    def create(self) -> BatchInputModule:
        return EyebrowMorphingCombiner00(self.args)


if __name__ == "__main__":
    cuda = torch.device('cuda')
    args = EyebrowMorphingCombiner00Args(
        image_size=128,
        image_channels=4,
        num_pose_params=12,
        start_channels=64,
        bottleneck_image_size=16,
        num_bottleneck_blocks=3,
        block_args=BlockArgs(
            initialization_method='xavier',
            use_spectral_norm=False,
            normalization_layer_factory=InstanceNorm2dFactory(),
            nonlinearity_factory=ReLUFactory(inplace=True)))
    face_morpher = EyebrowMorphingCombiner00(args).to(cuda)

    background_layer = torch.randn(8, 4, 128, 128, device=cuda)
    eyebrow_layer = torch.randn(8, 4, 128, 128, device=cuda)
    pose = torch.randn(8, 12, device=cuda)
    outputs = face_morpher.forward(background_layer, eyebrow_layer, pose)
    for i in range(len(outputs)):
        print(i, outputs[i].shape)
