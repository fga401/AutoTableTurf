import os
import unittest

from capture import FileLoader
from tableturf.debugger.cv import OpenCVDebugger
from tableturf.manager import detection

path = os.path.join(os.path.realpath(__file__), '..', '..', '..', '..', 'temp')


@unittest.skipIf(not os.path.exists(path), 'images not existed')
class TestStage(unittest.TestCase):

    def test_find_stage_roi_1(self):
        capture = FileLoader(path=os.path.join(path, 'stage1'))
        for _ in range(10):
            detection.stage_rois(capture.capture(), OpenCVDebugger())

    def test_find_stage_roi_2(self):
        capture = FileLoader(path=os.path.join(path, 'stage2'))
        for _ in range(10):
            detection.stage_rois(capture.capture(), OpenCVDebugger())

    def test_find_stage_roi_3(self):
        capture = FileLoader(path=os.path.join(path, 'stage3'))
        for _ in range(10):
            detection.stage_rois(capture.capture(), OpenCVDebugger())

    def test_stage_4(self):
        roi_capture = FileLoader(files=[os.path.join(path, 'stage4', 'roi.jpg')])
        capture = FileLoader(path=os.path.join(path, 'stage4'))
        rois, width, height = detection.stage_rois(roi_capture.capture())
        for _ in range(20):
            detection.stage(capture.capture(), rois, width, height, debug=OpenCVDebugger())

    def test_stage_5(self):
        roi_capture = FileLoader(files=[os.path.join(path, 'stage2', '0.jpg')])
        capture = FileLoader(path=os.path.join(path, 'stage5'))
        rois, width, height = detection.stage_rois(roi_capture.capture())
        for _ in range(20):
            detection.stage(capture.capture(), rois, width, height, debug=OpenCVDebugger())

    def test_preview_4(self):
        roi_capture = FileLoader(files=[os.path.join(path, 'stage4', 'roi.jpg')])
        stage_capture = FileLoader(files=[os.path.join(path, 'stage4', 'sp_off_10.jpg')])
        capture = FileLoader(path=os.path.join(path, 'stage4'))
        rois, width, height = detection.stage_rois(roi_capture.capture())
        for _ in range(30):
            detected_stage, is_fiery = detection.stage(stage_capture.capture(), rois, width, height)
            detection.preview(capture.capture(), detected_stage, is_fiery, rois, width, height, OpenCVDebugger())

    def test_preview_6(self):
        roi_capture = FileLoader(files=[os.path.join(path, 'stage1', 'i1.jpg')])
        stage_capture = FileLoader(files=[os.path.join(path, 'stage6', '10.jpg')])
        capture = FileLoader(path=os.path.join(path, 'stage6'))
        rois, width, height = detection.stage_rois(roi_capture.capture())
        for _ in range(30):
            detected_stage, is_fiery = detection.stage(stage_capture.capture(), rois, width, height, debug=None)
            detection.preview(capture.capture(), detected_stage, is_fiery, rois, width, height, OpenCVDebugger())

    def test_sp(self):
        capture = FileLoader(path=os.path.join(path, 'stage5'))
        for _ in range(10):
            detection.sp(capture.capture(), True)
