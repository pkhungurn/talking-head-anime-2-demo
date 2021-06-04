import errno
import json
import os
import socket
import sys
import threading
import time

sys.path.append(os.getcwd())

import numpy
import torch
import wx

from tha2.poser.poser import Poser
from tha2.mocap.ifacialmocap_constants import *
from tha2.mocap.ifacialmocap_pose_converter import IFacialMocapPoseConverter
from tha2.util import extract_PIL_image_from_filelike, resize_PIL_image, extract_pytorch_image_from_PIL_image, convert_output_image_from_torch_to_numpy


class CaptureData:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = self.create_default_data()

    def write_data(self, data):
        self.lock.acquire()
        self.data = data
        self.lock.release()

    def read_data(self):
        self.lock.acquire()
        output = self.data
        self.lock.release()
        return output

    @staticmethod
    def create_default_data():
        data = {}

        for blendshape_name in BLENDSHAPE_NAMES:
            data[blendshape_name] = 0.0

        data[HEAD_BONE_X] = 0.0
        data[HEAD_BONE_Y] = 0.0
        data[HEAD_BONE_Z] = 0.0
        data[HEAD_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]

        data[LEFT_EYE_BONE_X] = 0.0
        data[LEFT_EYE_BONE_Y] = 0.0
        data[LEFT_EYE_BONE_Z] = 0.0
        data[LEFT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]

        data[RIGHT_EYE_BONE_X] = 0.0
        data[RIGHT_EYE_BONE_Y] = 0.0
        data[RIGHT_EYE_BONE_Z] = 0.0
        data[RIGHT_EYE_BONE_QUAT] = [0.0, 0.0, 0.0, 1.0]

        return data


class ClientThread(threading.Thread):
    def __init__(self, capture_data: CaptureData):
        super().__init__()
        self.capture_data = capture_data
        self.should_terminate = False
        self.address = "0.0.0.0"
        self.port = 50002

    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.socket.bind((self.address, self.port))
        while True:
            if self.should_terminate:
                break
            try:
                socket_bytes = self.socket.recv(8192)
            except socket.error as e:
                err = e.args[0]
                if err == errno.EAGAIN or err == errno.EWOULDBLOCK:
                    continue
                else:
                    raise e
            socket_string = socket_bytes.decode("utf-8")
            blender_data = json.loads(socket_string)
            data = self.convert_from_blender_data(blender_data)
            self.capture_data.write_data(data)
        self.socket.close()

    @staticmethod
    def convert_from_blender_data(blender_data):
        data = {}

        shape_key_values = blender_data["shapeKeyValues"]
        for blendshape_name in BLENDSHAPE_NAMES:
            data[blendshape_name] = shape_key_values[blendshape_name]

        head_bone = blender_data["boneValues"]["armatures"]["bones"]["headBone"]
        data[HEAD_BONE_X] = head_bone["e_rx"]
        data[HEAD_BONE_Y] = head_bone["e_ry"]
        data[HEAD_BONE_Z] = head_bone["e_rz"]
        data[HEAD_BONE_QUAT] = [
            head_bone["q_rx"],
            head_bone["q_ry"],
            head_bone["q_rz"],
            head_bone["q_w"],
        ]

        right_eye_bone = blender_data["boneValues"]["armatures"]["bones"]["rightEyeBone"]
        data[RIGHT_EYE_BONE_X] = right_eye_bone["e_rx"]
        data[RIGHT_EYE_BONE_Y] = right_eye_bone["e_ry"]
        data[RIGHT_EYE_BONE_Z] = right_eye_bone["e_rz"]
        data[RIGHT_EYE_BONE_QUAT] = [
            right_eye_bone["q_rx"],
            right_eye_bone["q_ry"],
            right_eye_bone["q_rz"],
            right_eye_bone["q_w"],
        ]

        left_eye_bone = blender_data["boneValues"]["armatures"]["bones"]["leftEyeBone"]
        data[LEFT_EYE_BONE_X] = left_eye_bone["e_rx"]
        data[LEFT_EYE_BONE_Y] = left_eye_bone["e_ry"]
        data[LEFT_EYE_BONE_Z] = left_eye_bone["e_rz"]
        data[LEFT_EYE_BONE_QUAT] = [
            left_eye_bone["q_rx"],
            left_eye_bone["q_ry"],
            left_eye_bone["q_rz"],
            left_eye_bone["q_w"],
        ]

        return data


class MainFrame(wx.Frame):
    def __init__(self, poser: Poser, pose_converter: IFacialMocapPoseConverter, device: torch.device):
        super().__init__(None, wx.ID_ANY, "iFacialMocap Puppeteer")
        self.pose_converter = pose_converter
        self.poser = poser
        self.device = device
        self.capture_data = CaptureData()
        self.client_thread = ClientThread(self.capture_data)
        self.create_ui()
        self.Bind(wx.EVT_CLOSE, self.on_close)

        self.capture_timer = wx.Timer(self, wx.ID_ANY)
        self.Bind(wx.EVT_TIMER, self.update_capture_panel, id=self.capture_timer.GetId())

        self.animation_timer = wx.Timer(self, wx.ID_ANY)
        self.Bind(wx.EVT_TIMER, self.update_result_image_panel, id=self.animation_timer.GetId())

        self.wx_source_image = None
        self.torch_source_image = None
        self.last_pose = None

        self.client_thread.start()
        self.capture_timer.Start(33)
        self.animation_timer.Start(33)

        self.source_image_string = "Nothing yet!"

    def on_close(self, event: wx.Event):
        self.client_thread.should_terminate = True
        self.client_thread.join()
        self.capture_timer.Stop()
        self.animation_timer.Stop()
        event.Skip()

    def create_animation_panel(self, parent):
        self.animation_panel = wx.Panel(parent, style=wx.RAISED_BORDER)
        self.animation_panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.animation_panel.SetSizer(self.animation_panel_sizer)
        self.animation_panel.SetAutoLayout(1)

        if True:
            self.input_panel = wx.Panel(self.animation_panel, size=(256, 368), style=wx.SIMPLE_BORDER)
            self.input_panel_sizer = wx.BoxSizer(wx.VERTICAL)
            self.input_panel.SetSizer(self.input_panel_sizer)
            self.input_panel.SetAutoLayout(1)
            self.animation_panel_sizer.Add(self.input_panel, 0, wx.FIXED_MINSIZE)

            self.source_image_panel = wx.Panel(self.input_panel, size=(256, 256), style=wx.SIMPLE_BORDER)
            self.source_image_panel.Bind(wx.EVT_PAINT, self.paint_source_image_panel)
            self.input_panel_sizer.Add(self.source_image_panel, 0, wx.FIXED_MINSIZE)

            self.load_image_button = wx.Button(self.input_panel, wx.ID_ANY, "Load Image")
            self.input_panel_sizer.Add(self.load_image_button, 1, wx.EXPAND)
            self.load_image_button.Bind(wx.EVT_BUTTON, self.load_image)

            self.input_panel_sizer.Fit(self.input_panel)

        if True:
            self.pose_converter.init_pose_converter_panel(self.animation_panel)

        if True:
            self.animation_left_panel = wx.Panel(self.animation_panel, style=wx.SIMPLE_BORDER)
            self.animation_left_panel_sizer = wx.BoxSizer(wx.VERTICAL)
            self.animation_left_panel.SetSizer(self.animation_left_panel_sizer)
            self.animation_left_panel.SetAutoLayout(1)
            self.animation_panel_sizer.Add(self.animation_left_panel, 0, wx.EXPAND)

            self.result_image_panel = wx.Panel(self.animation_left_panel, size=(256, 256), style=wx.SIMPLE_BORDER)
            self.result_image_panel.Bind(wx.EVT_PAINT, self.paint_result_image_panel)
            self.animation_left_panel_sizer.Add(self.result_image_panel, 0, wx.FIXED_MINSIZE)

            self.output_index_choice = wx.Choice(self.animation_left_panel,
                                                 choices=[str(i) for i in range(self.poser.get_output_length())])
            self.output_index_choice.SetSelection(0)
            self.animation_left_panel_sizer.Add(self.output_index_choice, 0, wx.EXPAND)

            separator = wx.StaticLine(self.animation_left_panel, -1, size=(256, 5))
            self.animation_left_panel_sizer.Add(separator, 0, wx.EXPAND)

            background_text = wx.StaticText(self.animation_left_panel, label="--- Background ---",
                                            style=wx.ALIGN_CENTER)
            self.animation_left_panel_sizer.Add(background_text, 0, wx.EXPAND)

            self.output_background_choice = wx.Choice(
                self.animation_left_panel,
                choices=[
                    "TRANSPARENT",
                    "GREEN",
                    "BLUE",
                    "BLACK",
                    "WHITE"
                ])
            self.output_background_choice.SetSelection(0)
            self.animation_left_panel_sizer.Add(self.output_background_choice, 0, wx.EXPAND)

            separator = wx.StaticLine(self.animation_left_panel, -1, size=(256, 5))
            self.animation_left_panel_sizer.Add(separator, 0, wx.EXPAND)

            self.fps_text = wx.StaticText(self.animation_left_panel, label="")
            self.animation_left_panel_sizer.Add(self.fps_text, wx.SizerFlags().Border())

            self.animation_left_panel_sizer.Fit(self.animation_left_panel)

        self.animation_panel_sizer.Fit(self.animation_panel)

    def create_ui(self):
        self.main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.main_sizer)
        self.SetAutoLayout(1)

        self.capture_pose_lock = threading.Lock()

        self.create_animation_panel(self)
        self.main_sizer.Add(self.animation_panel, wx.SizerFlags(0).Expand().Border(wx.ALL, 5))

        self.create_capture_panel(self)
        self.main_sizer.Add(self.capture_panel, wx.SizerFlags(0).Expand().Border(wx.ALL, 5))

        self.main_sizer.Fit(self)

    def create_capture_panel(self, parent):
        self.capture_panel = wx.Panel(parent, style=wx.RAISED_BORDER)
        self.capture_panel_sizer = wx.FlexGridSizer(cols=5)
        for i in range(5):
            self.capture_panel_sizer.AddGrowableCol(i)
        self.capture_panel.SetSizer(self.capture_panel_sizer)
        self.capture_panel.SetAutoLayout(1)

        self.blendshape_labels = {}
        self.blendshape_value_labels = {}
        self.blendshape_gauges = {}
        blendshape_column_0 = self.create_blendshapes_column(self.capture_panel, COLUMN_0_BLENDSHAPES)
        self.capture_panel_sizer.Add(blendshape_column_0, wx.SizerFlags(0).Expand().Border(wx.ALL, 3))
        blendshape_column_1 = self.create_blendshapes_column(self.capture_panel, COLUMN_1_BLENDSHAPES)
        self.capture_panel_sizer.Add(blendshape_column_1, wx.SizerFlags(0).Expand().Border(wx.ALL, 3))
        blendshape_column_2 = self.create_blendshapes_column(self.capture_panel, COLUMN_2_BLENDSHAPES)
        self.capture_panel_sizer.Add(blendshape_column_2, wx.SizerFlags(0).Expand().Border(wx.ALL, 3))
        blendshape_column_3 = self.create_blendshapes_column(self.capture_panel, COLUMN_3_BLENDSHAPES)
        self.capture_panel_sizer.Add(blendshape_column_3, wx.SizerFlags(0).Expand().Border(wx.ALL, 3))
        blendshape_column_4 = self.create_blendshapes_column(self.capture_panel, COLUMN_4_BLENDSHAPES)
        self.capture_panel_sizer.Add(blendshape_column_4, wx.SizerFlags(0).Expand().Border(wx.ALL, 3))

        self.rotation_labels = {}
        self.rotation_value_labels = {}
        rotation_column_0 = self.create_rotation_column(self.capture_panel, RIGHT_EYE_BONE_ROTATIONS)
        self.capture_panel_sizer.Add(rotation_column_0, wx.SizerFlags(0).Expand().Border(wx.ALL, 3))
        rotation_column_1 = self.create_rotation_column(self.capture_panel, LEFT_EYE_BONE_ROTATIONS)
        self.capture_panel_sizer.Add(rotation_column_1, wx.SizerFlags(0).Expand().Border(wx.ALL, 3))
        rotation_column_2 = self.create_rotation_column(self.capture_panel, HEAD_BONE_ROTATIONS)
        self.capture_panel_sizer.Add(rotation_column_2, wx.SizerFlags(0).Expand().Border(wx.ALL, 3))

    def create_blendshapes_column(self, parent, blendshape_names):
        column_panel = wx.Panel(parent, style=wx.SIMPLE_BORDER)
        column_panel_sizer = wx.FlexGridSizer(cols=3)
        column_panel_sizer.AddGrowableCol(1)
        column_panel.SetSizer(column_panel_sizer)
        column_panel.SetAutoLayout(1)

        for blendshape_name in blendshape_names:
            self.blendshape_labels[blendshape_name] = wx.StaticText(
                column_panel, label=blendshape_name, style=wx.ALIGN_RIGHT)
            column_panel_sizer.Add(self.blendshape_labels[blendshape_name],
                                   wx.SizerFlags(1).Expand().Border(wx.ALL, 3))

            self.blendshape_gauges[blendshape_name] = wx.Gauge(
                column_panel, style=wx.GA_HORIZONTAL, size=(100, -1))
            column_panel_sizer.Add(self.blendshape_gauges[blendshape_name], wx.SizerFlags(1).Expand().Border(wx.ALL, 3))

            self.blendshape_value_labels[blendshape_name] = wx.TextCtrl(
                column_panel, style=wx.TE_RIGHT, size=(40, -1))
            self.blendshape_value_labels[blendshape_name].SetValue("0.00")
            self.blendshape_value_labels[blendshape_name].Disable()
            column_panel_sizer.Add(self.blendshape_value_labels[blendshape_name],
                                   wx.SizerFlags(0).Border(wx.ALL, 3))

        column_panel.GetSizer().Fit(column_panel)
        return column_panel

    def create_rotation_column(self, parent, rotation_names):
        column_panel = wx.Panel(parent, style=wx.SIMPLE_BORDER)
        column_panel_sizer = wx.FlexGridSizer(cols=2)
        column_panel_sizer.AddGrowableCol(1)
        column_panel.SetSizer(column_panel_sizer)
        column_panel.SetAutoLayout(1)

        for rotation_name in rotation_names:
            self.rotation_labels[rotation_name] = wx.StaticText(
                column_panel, label=rotation_name, style=wx.ALIGN_RIGHT)
            column_panel_sizer.Add(self.rotation_labels[rotation_name],
                                   wx.SizerFlags(1).Expand().Border(wx.ALL, 3))

            self.rotation_value_labels[rotation_name] = wx.TextCtrl(
                column_panel, style=wx.TE_RIGHT)
            self.rotation_value_labels[rotation_name].SetValue("0.00")
            self.rotation_value_labels[rotation_name].Disable()
            column_panel_sizer.Add(self.rotation_value_labels[rotation_name],
                                   wx.SizerFlags(1).Expand().Border(wx.ALL, 3))

        column_panel.GetSizer().Fit(column_panel)
        return column_panel

    def paint_capture_panel(self, event: wx.Event):
        self.update_capture_panel(event)

    def update_capture_panel(self, event: wx.Event):
        data = self.capture_data.read_data()
        for blendshape_name in BLENDSHAPE_NAMES:
            value = data[blendshape_name]
            self.blendshape_gauges[blendshape_name].SetValue(MainFrame.convert_to_100(value))
            self.blendshape_value_labels[blendshape_name].SetValue("%0.2f" % value)
        for rotation_name in ROTATION_NAMES:
            value = data[rotation_name]
            self.rotation_value_labels[rotation_name].SetValue("%0.2f" % value)

    @staticmethod
    def convert_to_100(x):
        return int(max(0.0, min(1.0, x)) * 100)

    def paint_source_image_panel(self, event: wx.Event):
        if self.wx_source_image is None:
            self.draw_source_image_string(self.source_image_panel, use_paint_dc=True)
        else:
            dc = wx.PaintDC(self.source_image_panel)
            dc.Clear()
            dc.DrawBitmap(self.wx_source_image, 0, 0, True)

    def draw_source_image_string(self, widget, use_paint_dc: bool = True):
        if use_paint_dc:
            dc = wx.PaintDC(widget)
        else:
            dc = wx.ClientDC(widget)
        dc.Clear()
        font = wx.Font(wx.FontInfo(14).Family(wx.FONTFAMILY_SWISS))
        dc.SetFont(font)
        w, h = dc.GetTextExtent(self.source_image_string)
        dc.DrawText(self.source_image_string, 128 - w // 2, 128 - h // 2)

    def paint_result_image_panel(self, event: wx.Event):
        self.last_pose = None

    def update_result_image_panel(self, event: wx.Event):
        tic = time.perf_counter()

        ifacialmocap_pose = self.capture_data.read_data()
        current_pose = self.pose_converter.convert(ifacialmocap_pose)
        if self.last_pose is not None \
                and self.last_pose == current_pose \
                and self.last_output_index == self.output_index_choice.GetSelection():
            return
        self.last_pose = current_pose
        self.last_output_index = self.output_index_choice.GetSelection()

        if self.torch_source_image is None:
            self.draw_source_image_string(self.result_image_panel, use_paint_dc=False)
            return

        pose = torch.tensor(current_pose, device=self.device)
        output_index = self.output_index_choice.GetSelection()
        output_image = self.poser.pose(self.torch_source_image, pose, output_index)[0].detach().cpu()
        numpy_image = convert_output_image_from_torch_to_numpy(output_image)

        background_choice = self.output_background_choice.GetSelection()
        if background_choice != 0:
            background = numpy.zeros((numpy_image.shape[0], numpy_image.shape[1], numpy_image.shape[2]))
            background[:, :, 3] = 1.0
            if background_choice == 1:
                background[:, :, 1] = 1.0
            elif background_choice == 2:
                background[:, :, 2] = 1.0
            elif background_choice == 4:
                background[:, :, 0:3] = 1.0
            numpy_image = self.blend_with_background(numpy_image, background)

        numpy_image = numpy.uint8(numpy.rint(numpy_image * 255.0))
        wx_image = wx.ImageFromBuffer(
            numpy_image.shape[0],
            numpy_image.shape[1],
            numpy_image[:, :, 0:3].tobytes(),
            numpy_image[:, :, 3].tobytes())
        wx_bitmap = wx_image.ConvertToBitmap()

        dc = wx.ClientDC(self.result_image_panel)
        dc.Clear()
        dc.DrawBitmap(wx_bitmap, (256 - numpy_image.shape[0]) // 2, (256 - numpy_image.shape[1]) // 2, True)

        toc = time.perf_counter()
        elapsed_time = toc - tic
        fps = min(1.0 / elapsed_time, 1000.0 / 33.0)
        self.fps_text.SetLabelText("FPS = %0.2f" % fps)

    def blend_with_background(self, numpy_image, background):
        alpha = numpy_image[:, :, 3:4]
        color = numpy_image[:, :, 0:3]
        new_color = color * alpha + (1.0 - alpha) * background[:, :, 0:3]
        return numpy.concatenate([new_color, background[:, :, 3:4]], axis=2)

    def load_image(self, event: wx.Event):
        dir_name = "data/illust"
        file_dialog = wx.FileDialog(self, "Choose an image", dir_name, "", "*.png", wx.FD_OPEN)
        if file_dialog.ShowModal() == wx.ID_OK:
            image_file_name = os.path.join(file_dialog.GetDirectory(), file_dialog.GetFilename())
            pil_image = resize_PIL_image(extract_PIL_image_from_filelike(image_file_name))
            w, h = pil_image.size
            if pil_image.mode != 'RGBA':
                self.source_image_string = "Image must have alpha channel!"
                self.wx_source_image = None
                self.torch_source_image = None
            else:
                self.wx_source_image = wx.Bitmap.FromBufferRGBA(w, h, pil_image.convert("RGBA").tobytes())
                self.torch_source_image = extract_pytorch_image_from_PIL_image(pil_image).to(self.device)

            self.Refresh()
        file_dialog.Destroy()


if __name__ == "__main__":
    import tha2.poser.modes.mode_20
    import tha2.poser.modes.mode_20_wx

    cuda = torch.device('cuda')
    poser = tha2.poser.modes.mode_20.create_poser(cuda)
    pose_converter = tha2.poser.modes.mode_20_wx.create_ifacialmocap_pose_converter()

    app = wx.App()
    main_frame = MainFrame(poser, pose_converter, cuda)
    main_frame.Show(True)
    app.MainLoop()
