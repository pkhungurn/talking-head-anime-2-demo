import math
from enum import Enum
from typing import List, Dict, Optional

import numpy
import scipy.optimize
import wx

from tha2.mocap.ifacialmocap_constants import *
from tha2.mocap.ifacialmocap_pose_converter import IFacialMocapPoseConverter
from tha2.poser.modes.mode_20 import get_pose_parameters


def clamp(x, min_value, max_value):
    return max(min_value, min(max_value, x))


class EyebrowDownMode(Enum):
    TROUBLED = 1
    ANGRY = 2
    LOWERED = 3
    SERIOUS = 4


class WinkMode(Enum):
    NORMAL = 1
    RELAXED = 2


class IFacialMocapPoseConverter20Args:
    def __init__(self,
                 lower_smile_threshold: float = 0.4,
                 upper_smile_threshold: float = 0.6,
                 eyebrow_down_mode: EyebrowDownMode = EyebrowDownMode.ANGRY,
                 wink_mode: WinkMode = WinkMode.NORMAL,
                 eye_surprised_max_value: float = 0.5,
                 eye_wink_max_value: float = 0.8,
                 eyebrow_down_max_value: float = 0.4,
                 cheek_squint_min_value: float = 0.1,
                 cheek_squint_max_value: float = 0.7,
                 eye_rotation_factor: float = 1.0 / 0.75,
                 jaw_open_min_value: float = 0.1,
                 jaw_open_max_value: float = 0.4,
                 mouth_frown_max_value: float = 0.6,
                 mouth_funnel_min_value: float = 0.25,
                 mouth_funnel_max_value: float = 0.5,
                 iris_small_left=0.0,
                 iris_small_right=0.0):
        self.iris_small_right = iris_small_left
        self.iris_small_left = iris_small_right
        self.wink_mode = wink_mode
        self.mouth_funnel_max_value = mouth_funnel_max_value
        self.mouth_funnel_min_value = mouth_funnel_min_value
        self.mouth_frown_max_value = mouth_frown_max_value
        self.jaw_open_max_value = jaw_open_max_value
        self.jaw_open_min_value = jaw_open_min_value
        self.eye_rotation_factor = eye_rotation_factor
        self.cheek_squint_max_value = cheek_squint_max_value
        self.cheek_squint_min_value = cheek_squint_min_value
        self.eyebrow_down_max_value = eyebrow_down_max_value
        self.eye_blink_max_value = eye_wink_max_value
        self.eye_wide_max_value = eye_surprised_max_value
        self.eyebrow_down_mode = eyebrow_down_mode
        self.lower_smile_threshold = lower_smile_threshold
        self.upper_smile_threshold = upper_smile_threshold


class IFacialMocapPoseConverter20(IFacialMocapPoseConverter):
    def __init__(self, args: Optional[IFacialMocapPoseConverter20Args] = None):
        super().__init__()
        if args is None:
            args = IFacialMocapPoseConverter20Args()
        self.args = args
        pose_parameters = get_pose_parameters()
        self.pose_size = 42

        self.eyebrow_troubled_left_index = pose_parameters.get_parameter_index("eyebrow_troubled_left")
        self.eyebrow_troubled_right_index = pose_parameters.get_parameter_index("eyebrow_troubled_right")
        self.eyebrow_angry_left_index = pose_parameters.get_parameter_index("eyebrow_angry_left")
        self.eyebrow_angry_right_index = pose_parameters.get_parameter_index("eyebrow_angry_right")
        self.eyebrow_happy_left_index = pose_parameters.get_parameter_index("eyebrow_happy_left")
        self.eyebrow_happy_right_index = pose_parameters.get_parameter_index("eyebrow_happy_right")
        self.eyebrow_raised_left_index = pose_parameters.get_parameter_index("eyebrow_raised_left")
        self.eyebrow_raised_right_index = pose_parameters.get_parameter_index("eyebrow_raised_right")
        self.eyebrow_lowered_left_index = pose_parameters.get_parameter_index("eyebrow_lowered_left")
        self.eyebrow_lowered_right_index = pose_parameters.get_parameter_index("eyebrow_lowered_right")
        self.eyebrow_serious_left_index = pose_parameters.get_parameter_index("eyebrow_serious_left")
        self.eyebrow_serious_right_index = pose_parameters.get_parameter_index("eyebrow_serious_right")

        self.eye_surprised_left_index = pose_parameters.get_parameter_index("eye_surprised_left")
        self.eye_surprised_right_index = pose_parameters.get_parameter_index("eye_surprised_right")
        self.eye_wink_left_index = pose_parameters.get_parameter_index("eye_wink_left")
        self.eye_wink_right_index = pose_parameters.get_parameter_index("eye_wink_right")
        self.eye_happy_wink_left_index = pose_parameters.get_parameter_index("eye_happy_wink_left")
        self.eye_happy_wink_right_index = pose_parameters.get_parameter_index("eye_happy_wink_right")
        self.eye_relaxed_left_index = pose_parameters.get_parameter_index("eye_relaxed_left")
        self.eye_relaxed_right_index = pose_parameters.get_parameter_index("eye_relaxed_right")
        self.eye_raised_lower_eyelid_left_index = pose_parameters.get_parameter_index("eye_raised_lower_eyelid_left")
        self.eye_raised_lower_eyelid_right_index = pose_parameters.get_parameter_index("eye_raised_lower_eyelid_right")

        self.iris_small_left_index = pose_parameters.get_parameter_index("iris_small_left")
        self.iris_small_right_index = pose_parameters.get_parameter_index("iris_small_right")

        self.iris_rotation_x_index = pose_parameters.get_parameter_index("iris_rotation_x")
        self.iris_rotation_y_index = pose_parameters.get_parameter_index("iris_rotation_y")

        self.head_x_index = pose_parameters.get_parameter_index("head_x")
        self.head_y_index = pose_parameters.get_parameter_index("head_y")
        self.neck_z_index = pose_parameters.get_parameter_index("neck_z")

        self.mouth_aaa_index = pose_parameters.get_parameter_index("mouth_aaa")
        self.mouth_iii_index = pose_parameters.get_parameter_index("mouth_iii")
        self.mouth_uuu_index = pose_parameters.get_parameter_index("mouth_uuu")
        self.mouth_eee_index = pose_parameters.get_parameter_index("mouth_eee")
        self.mouth_ooo_index = pose_parameters.get_parameter_index("mouth_ooo")

        self.mouth_lowered_corner_left_index = pose_parameters.get_parameter_index("mouth_lowered_corner_left")
        self.mouth_lowered_corner_right_index = pose_parameters.get_parameter_index("mouth_lowered_corner_right")
        self.mouth_raised_corner_left_index = pose_parameters.get_parameter_index("mouth_raised_corner_left")
        self.mouth_raised_corner_right_index = pose_parameters.get_parameter_index("mouth_raised_corner_right")

    def convert(self, ifacialmocap_pose: Dict[str, float]) -> List[float]:
        pose = [0.0 for i in range(self.pose_size)]

        smile_value = \
            (ifacialmocap_pose[MOUTH_SMILE_LEFT] + ifacialmocap_pose[MOUTH_SMILE_RIGHT]) / 2.0 \
            + ifacialmocap_pose[MOUTH_SHRUG_UPPER]
        smile_degree = clamp((smile_value - self.args.lower_smile_threshold) / (
                self.args.upper_smile_threshold - self.args.lower_smile_threshold), 0.0, 1.0)

        # Eyebrow
        if True:
            brow_inner_up = ifacialmocap_pose[BROW_INNER_UP]
            brow_outer_up_right = ifacialmocap_pose[BROW_OUTER_UP_RIGHT]
            brow_outer_up_left = ifacialmocap_pose[BROW_OUTER_UP_LEFT]

            brow_up_left = clamp(brow_inner_up + brow_outer_up_left, 0.0, 1.0)
            brow_up_right = clamp(brow_inner_up + brow_outer_up_right, 0.0, 1.0)
            pose[self.eyebrow_raised_left_index] = brow_up_left
            pose[self.eyebrow_raised_right_index] = brow_up_right

            brow_down_left = (1.0 - smile_degree) \
                             * clamp(ifacialmocap_pose[BROW_DOWN_LEFT] / self.args.eyebrow_down_max_value, 0.0, 1.0)
            brow_down_right = (1.0 - smile_degree) \
                              * clamp(ifacialmocap_pose[BROW_DOWN_RIGHT] / self.args.eyebrow_down_max_value, 0.0, 1.0)
            if self.args.eyebrow_down_mode == EyebrowDownMode.TROUBLED:
                pose[self.eyebrow_troubled_left_index] = brow_down_left
                pose[self.eyebrow_troubled_right_index] = brow_down_right
            elif self.args.eyebrow_down_mode == EyebrowDownMode.ANGRY:
                pose[self.eyebrow_angry_left_index] = brow_down_left
                pose[self.eyebrow_angry_right_index] = brow_down_right
            elif self.args.eyebrow_down_mode == EyebrowDownMode.LOWERED:
                pose[self.eyebrow_lowered_left_index] = brow_down_left
                pose[self.eyebrow_lowered_right_index] = brow_down_right
            elif self.args.eyebrow_down_mode == EyebrowDownMode.SERIOUS:
                pose[self.eyebrow_serious_left_index] = brow_down_left
                pose[self.eyebrow_serious_right_index] = brow_down_right

            brow_happy_value = clamp(smile_value, 0.0, 1.0) * smile_degree
            pose[self.eyebrow_happy_left_index] = brow_happy_value
            pose[self.eyebrow_happy_right_index] = brow_happy_value

        # Eye
        if True:
            # Surprised
            pose[self.eye_surprised_left_index] = clamp(
                ifacialmocap_pose[EYE_WIDE_LEFT] / self.args.eye_wide_max_value, 0.0, 1.0)
            pose[self.eye_surprised_right_index] = clamp(
                ifacialmocap_pose[EYE_WIDE_RIGHT] / self.args.eye_wide_max_value, 0.0, 1.0)

            # Wink
            if self.args.wink_mode == WinkMode.NORMAL:
                wink_left_index = self.eye_wink_left_index
                wink_right_index = self.eye_wink_right_index
            else:
                wink_left_index = self.eye_relaxed_left_index
                wink_right_index = self.eye_relaxed_right_index
            pose[wink_left_index] = (1.0 - smile_degree) * clamp(
                ifacialmocap_pose[EYE_BLINK_LEFT] / self.args.eye_blink_max_value, 0.0, 1.0)
            pose[wink_right_index] = (1.0 - smile_degree) * clamp(
                ifacialmocap_pose[EYE_BLINK_RIGHT] / self.args.eye_blink_max_value, 0.0, 1.0)
            pose[self.eye_happy_wink_left_index] = smile_degree * clamp(
                ifacialmocap_pose[EYE_BLINK_LEFT] / self.args.eye_blink_max_value, 0.0, 1.0)
            pose[self.eye_happy_wink_right_index] = smile_degree * clamp(
                ifacialmocap_pose[EYE_BLINK_RIGHT] / self.args.eye_blink_max_value, 0.0, 1.0)

            # Lower eyelid
            cheek_squint_denom = self.args.cheek_squint_max_value - self.args.cheek_squint_min_value
            pose[self.eye_raised_lower_eyelid_left_index] = \
                clamp(
                    (ifacialmocap_pose[CHEEK_SQUINT_LEFT] - self.args.cheek_squint_min_value) / cheek_squint_denom,
                    0.0, 1.0)
            pose[self.eye_raised_lower_eyelid_right_index] = \
                clamp(
                    (ifacialmocap_pose[CHEEK_SQUINT_RIGHT] - self.args.cheek_squint_min_value) / cheek_squint_denom,
                    0.0, 1.0)

        # Iris rotation
        if True:
            eye_rotation_y = (ifacialmocap_pose[EYE_LOOK_IN_LEFT] -
                              ifacialmocap_pose[EYE_LOOK_OUT_LEFT] -
                              ifacialmocap_pose[EYE_LOOK_IN_RIGHT] +
                              ifacialmocap_pose[EYE_LOOK_OUT_RIGHT]) / 2.0 * self.args.eye_rotation_factor
            pose[self.iris_rotation_y_index] = clamp(eye_rotation_y, -1.0, 1.0)

            eye_rotation_x = (ifacialmocap_pose[EYE_LOOK_UP_LEFT]
                              + ifacialmocap_pose[EYE_LOOK_UP_RIGHT]
                              - ifacialmocap_pose[EYE_LOOK_DOWN_RIGHT]
                              + ifacialmocap_pose[EYE_LOOK_DOWN_LEFT]) / 2.0 * self.args.eye_rotation_factor
            pose[self.iris_rotation_x_index] = clamp(eye_rotation_x, -1.0, 1.0)

        # Iris size
        if True:
            pose[self.iris_small_left_index] = self.args.iris_small_left
            pose[self.iris_small_right_index] = self.args.iris_small_right

        # Head rotation
        if True:
            pose[self.head_x_index] = clamp(-ifacialmocap_pose[HEAD_BONE_X] * 180.0 / math.pi, -15.0, 15.0) / 15.0
            pose[self.head_y_index] = clamp(-ifacialmocap_pose[HEAD_BONE_Y] * 180.0 / math.pi, -15.0, 15.0) / 15.0
            pose[self.neck_z_index] = clamp(ifacialmocap_pose[HEAD_BONE_Z] * 180.0 / math.pi, -15.0, 15.0) / 15.0

        # Mouth
        if True:
            # mouth_open = clamp((ifacialmocap_pose[JAW_OPEN] - 0.10) / 0.7, 0.0, 1.0)
            jaw_open_denom = self.args.jaw_open_max_value - self.args.jaw_open_min_value
            mouth_open = clamp((ifacialmocap_pose[JAW_OPEN] - self.args.jaw_open_min_value) / jaw_open_denom, 0.0, 1.0)
            pose[self.mouth_aaa_index] = mouth_open
            pose[self.mouth_raised_corner_left_index] = clamp(smile_value, 0.0, 1.0)
            pose[self.mouth_raised_corner_right_index] = clamp(smile_value, 0.0, 1.0)

            is_mouth_open = mouth_open > 0.0
            if not is_mouth_open:
                mouth_frown_value = clamp(
                    (ifacialmocap_pose[MOUTH_FROWN_LEFT] + ifacialmocap_pose[
                        MOUTH_FROWN_RIGHT]) / self.args.mouth_frown_max_value, 0.0, 1.0)
                pose[self.mouth_lowered_corner_left_index] = mouth_frown_value
                pose[self.mouth_lowered_corner_right_index] = mouth_frown_value
            else:
                mouth_lower_down = clamp(
                    ifacialmocap_pose[MOUTH_LOWER_DOWN_LEFT] + ifacialmocap_pose[MOUTH_LOWER_DOWN_RIGHT], 0.0, 1.0)
                mouth_funnel = ifacialmocap_pose[MOUTH_FUNNEL]
                mouth_pucker = ifacialmocap_pose[MOUTH_PUCKER]

                mouth_point = [mouth_open, mouth_lower_down, mouth_funnel, mouth_pucker]

                aaa_point = [1.0, 1.0, 0.0, 0.0]
                iii_point = [0.0, 1.0, 0.0, 0.0]
                uuu_point = [0.5, 0.3, 0.25, 0.75]
                ooo_point = [1.0, 0.5, 0.5, 0.4]

                decomp = numpy.array([0, 0, 0, 0])
                M = numpy.array([
                    aaa_point,
                    iii_point,
                    uuu_point,
                    ooo_point
                ])

                def loss(decomp):
                    return numpy.linalg.norm(numpy.matmul(decomp, M) - mouth_point) \
                           + 0.01 * numpy.linalg.norm(decomp, ord=1)

                opt_result = scipy.optimize.minimize(loss, decomp,
                                                     bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
                decomp = opt_result["x"]
                restricted_decomp = [decomp.item(0), decomp.item(1), decomp.item(2), decomp.item(3)]
                # restricted_decomp = restrict_to_two_morphs(
                #    [decomp.item(0), decomp.item(1), decomp.item(2), decomp.item(3)])
                pose[self.mouth_aaa_index] = restricted_decomp[0]
                pose[self.mouth_iii_index] = restricted_decomp[1]
                mouth_funnel_denom = self.args.mouth_funnel_max_value - self.args.mouth_funnel_min_value
                ooo_alpha = clamp((mouth_funnel - self.args.mouth_funnel_min_value) / mouth_funnel_denom, 0.0, 1.0)
                uo_value = clamp(restricted_decomp[2] + restricted_decomp[3], 0.0, 1.0)
                pose[self.mouth_uuu_index] = uo_value * (1.0 - ooo_alpha)
                pose[self.mouth_ooo_index] = uo_value * ooo_alpha
                # pose[28] = restricted_decomp[2]
                # pose[30] = restricted_decomp[3]

        return pose

    def init_pose_converter_panel(self, parent):
        self.panel = wx.Panel(parent, style=wx.SIMPLE_BORDER)
        self.panel_sizer = wx.BoxSizer(wx.VERTICAL)
        self.panel.SetSizer(self.panel_sizer)
        self.panel.SetAutoLayout(1)
        parent.GetSizer().Add(self.panel, 0, wx.EXPAND)

        if True:
            eyebrow_down_mode_text = wx.StaticText(self.panel, label=" --- Eyebrow Down Mode --- ",
                                                   style=wx.ALIGN_CENTER)
            self.panel_sizer.Add(eyebrow_down_mode_text, 0, wx.EXPAND)

            self.eyebrow_down_mode_choice = wx.Choice(
                self.panel,
                choices=[
                    "ANGRY",
                    "TROUBLED",
                    "SERIOUS",
                    "LOWERED",
                ])
            self.eyebrow_down_mode_choice.SetSelection(0)
            self.panel_sizer.Add(self.eyebrow_down_mode_choice, 0, wx.EXPAND)
            self.eyebrow_down_mode_choice.Bind(wx.EVT_CHOICE, self.change_eyebrow_down_mode)

            separator = wx.StaticLine(self.panel, -1, size=(256, 5))
            self.panel_sizer.Add(separator, 0, wx.EXPAND)

        if True:
            wink_mode_text = wx.StaticText(self.panel, label=" --- Wink Mode --- ", style=wx.ALIGN_CENTER)
            self.panel_sizer.Add(wink_mode_text, 0, wx.EXPAND)

            self.wink_mode_choice = wx.Choice(
                self.panel,
                choices=[
                    "NORMAL",
                    "RELAXED",
                ])
            self.wink_mode_choice.SetSelection(0)
            self.panel_sizer.Add(self.wink_mode_choice, 0, wx.EXPAND)
            self.wink_mode_choice.Bind(wx.EVT_CHOICE, self.change_wink_mode)

            separator = wx.StaticLine(self.panel, -1, size=(256, 5))
            self.panel_sizer.Add(separator, 0, wx.EXPAND)

        if True:
            iris_size_text = wx.StaticText(self.panel, label=" --- Iris Size --- ", style=wx.ALIGN_CENTER)
            self.panel_sizer.Add(iris_size_text, 0, wx.EXPAND)

            self.iris_left_slider = wx.Slider(self.panel, minValue=0, maxValue=1000, value=0, style=wx.HORIZONTAL)
            self.panel_sizer.Add(self.iris_left_slider, 0, wx.EXPAND)
            self.iris_left_slider.Bind(wx.EVT_SLIDER, self.change_iris_size)

            self.iris_right_slider = wx.Slider(self.panel, minValue=0, maxValue=1000, value=0, style=wx.HORIZONTAL)
            self.panel_sizer.Add(self.iris_right_slider, 0, wx.EXPAND)
            self.iris_right_slider.Bind(wx.EVT_SLIDER, self.change_iris_size)
            self.iris_right_slider.Enable(False)

            self.link_left_right_irises = wx.CheckBox(
                self.panel, label="Use same value for both sides")
            self.link_left_right_irises.SetValue(True)
            self.panel_sizer.Add(self.link_left_right_irises, wx.SizerFlags().CenterHorizontal().Border())
            self.link_left_right_irises.Bind(wx.EVT_CHECKBOX, self.link_left_right_irises_clicked)

        self.panel_sizer.Fit(self.panel)

    def change_eyebrow_down_mode(self, event: wx.Event):
        selected_index = self.eyebrow_down_mode_choice.GetSelection()
        if selected_index == 0:
            self.args.eyebrow_down_mode = EyebrowDownMode.ANGRY
        elif selected_index == 1:
            self.args.eyebrow_down_mode = EyebrowDownMode.TROUBLED
        elif selected_index == 2:
            self.args.eyebrow_down_mode = EyebrowDownMode.SERIOUS
        else:
            self.args.eyebrow_down_mode = EyebrowDownMode.LOWERED

    def change_wink_mode(self, event: wx.Event):
        selected_index = self.wink_mode_choice.GetSelection()
        if selected_index == 0:
            self.args.wink_mode = WinkMode.NORMAL
        else:
            self.args.wink_mode = WinkMode.RELAXED

    def change_iris_size(self, event: wx.Event):
        if self.link_left_right_irises.GetValue():
            left_value = self.iris_left_slider.GetValue()
            right_value = self.iris_right_slider.GetValue()
            if left_value != right_value:
                self.iris_right_slider.SetValue(left_value)
            self.args.iris_small_left = left_value / 1000.0
            self.args.iris_small_right = left_value / 1000.0
        else:
            self.args.iris_small_left = self.iris_left_slider.GetValue() / 1000.0
            self.args.iris_small_right = self.iris_right_slider.GetValue() / 1000.0

    def link_left_right_irises_clicked(self, event: wx.Event):
        if self.link_left_right_irises.GetValue():
            self.iris_right_slider.Enable(False)
        else:
            self.iris_right_slider.Enable(True)
        self.change_iris_size(event)


def create_ifacialmocap_pose_converter(
        args: Optional[IFacialMocapPoseConverter20Args] = None) -> IFacialMocapPoseConverter:
    return IFacialMocapPoseConverter20(args)
