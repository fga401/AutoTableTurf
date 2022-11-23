import os
import unittest

from capture import FileLoader
from tableturf.debugger.cv import OpenCVDebugger
from tableturf.manager import detection

path = os.path.join(os.path.realpath(__file__), '..', '..', '..', '..', 'temp')


@unittest.skipIf(not os.path.exists(path), 'images not existed')
class TestCard(unittest.TestCase):
    def test_hands_cursor(self):
        capture = FileLoader(path=os.path.join(path, 'stage2'))
        for _ in range(10):
            detection.hands_cursor(capture.capture(), OpenCVDebugger())

    def test_hands_in_battle(self):
        capture = FileLoader(path=os.path.join(path, 'stage2'))
        for _ in range(10):
            detection.hands(capture.capture(), None, OpenCVDebugger())

    def test_hands_in_redraw(self):
        capture = FileLoader(path=os.path.join(path, 'redraw'))
        for _ in range(10):
            detection.hands(capture.capture(), None, OpenCVDebugger())
