import torch
from torch import Tensor
from torch.nn import Sequential, Tanh, Sigmoid
from torch.nn.functional import affine_grid, grid_sample

from tha2.nn.backcomp.nn.conv import Conv7
from tha2.nn.backcomp.nn.encoder_decoder_module import EncoderDecoderModule
from tha2.nn.batch_module.batch_input_module import BatchInputModule, BatchInputModuleFactory


class TwoAlgoFaceRotator(BatchInputModule):
    COLOR_CHANGED_IMAGE_INDEX = 0
    RESAMPLED_IMAGE_INDEX = 1
    COLOR_CHANGE_INDEX = 2
    ALPHA_INDEX = 3
    GRID_CHANGE_INDEX = 4
    OUTPUT_LENGTH = 5

    def __init__(self,
                 image_size: int = 256,
                 image_channels: int = 4,
                 pose_size: int = 3,
                 intermediate_channels: int = 64,
                 bottleneck_image_size: int = 32,
                 bottleneck_block_count: int = 6,
                 initialization_method: str = 'he',
                 align_corners: bool = True):
        super().__init__()
        self.align_corners = align_corners
        self.main_body = EncoderDecoderModule(
            image_size=image_size,
            image_channels=image_channels + pose_size,
            output_channels=intermediate_channels,
            bottleneck_image_size=bottleneck_image_size,
            bottleneck_block_count=bottleneck_block_count,
            initialization_method=initialization_method)
        self.pumarola_color_change = Sequential(
            Conv7(intermediate_channels, image_channels, initialization_method),
            Tanh())
        self.pumarola_alpha_mask = Sequential(
            Conv7(intermediate_channels, image_channels, initialization_method),
            Sigmoid())
        self.zhou_grid_change = Conv7(intermediate_channels, 2, initialization_method)

    def forward(self, image: Tensor, pose: Tensor):
        n, c, h, w = image.shape

        pose = pose.unsqueeze(2).unsqueeze(3)
        pose = pose.expand(pose.shape[0], pose.shape[1], image.shape[2], image.shape[3])
        x = torch.cat([image, pose], dim=1)
        y = self.main_body(x)

        color_change = self.pumarola_color_change(y)
        alpha_mask = self.pumarola_alpha_mask(y)
        color_changed = alpha_mask * image + (1 - alpha_mask) * color_change

        original_grid_change = self.zhou_grid_change(y)
        grid_change = torch.transpose(original_grid_change.view(n, 2, h * w), 1, 2).view(n, h, w, 2)
        device = self.zhou_grid_change.weight.device
        identity = torch.Tensor([[1, 0, 0], [0, 1, 0]]).to(device).unsqueeze(0).repeat(n, 1, 1)
        base_grid = affine_grid(identity, [n, c, h, w], align_corners=self.align_corners)
        grid = base_grid + grid_change
        resampled = grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=self.align_corners)

        return [color_changed, resampled, color_change, alpha_mask, original_grid_change]

    def forward_from_batch(self, batch):
        return self.forward(batch[0], batch[1])


class TwoAlgoFaceRotatorFactory(BatchInputModuleFactory):
    def __init__(self,
                 image_size: int = 256,
                 image_channels: int = 4,
                 pose_size: int = 3,
                 intermediate_channels: int = 64,
                 bottleneck_image_size: int = 32,
                 bottleneck_block_count: int = 6,
                 initialization_method: str = 'he',
                 align_corners: bool = True):
        super().__init__()
        self.image_size = image_size
        self.image_channels = image_channels
        self.pose_size = pose_size
        self.intermediate_channels = intermediate_channels
        self.bottleneck_image_size = bottleneck_image_size
        self.bottleneck_block_count = bottleneck_block_count
        self.initialization_method = initialization_method
        self.align_corners = align_corners

    def create(self) -> BatchInputModule:
        return TwoAlgoFaceRotator(
            self.image_size,
            self.image_channels,
            self.pose_size,
            self.intermediate_channels,
            self.bottleneck_image_size,
            self.bottleneck_block_count,
            self.initialization_method,
            self.align_corners)