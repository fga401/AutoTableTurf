import os
import unittest

from capture import FileLoader
from tableturf.manager import detection

path = os.path.join(os.path.realpath(__file__), '..', '..', '..', '..', 'temp')


@unittest.skipIf(not os.path.exists(path), 'images not existed')
class TestCard(unittest.TestCase):

    def test_find_stage_roi_1(self):
        capture = FileLoader(path=os.path.join(path, 'stage1'))
        for _ in range(10):
            detection.stage_rois(capture.capture(), True)

    def test_find_stage_roi_2(self):
        capture = FileLoader(path=os.path.join(path, 'stage2'))
        for _ in range(10):
            detection.stage_rois(capture.capture(), True)

    def test_find_stage_roi_3(self):
        capture = FileLoader(path=os.path.join(path, 'stage3'))
        for _ in range(10):
            detection.stage_rois(capture.capture(), True)
