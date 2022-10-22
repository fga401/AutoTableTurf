import os
import unittest

from capture import FileLoader
from tableturf.manager import detection

path = os.path.join(os.path.realpath(__file__), '..', '..', '..', '..', 'temp', 'redraw')


@unittest.skipIf(not os.path.exists(path), 'images not existed')
class TestRedrawCursor(unittest.TestCase):
    def test_redraw_cursor(self):
        capture = FileLoader(path=path)
        for _ in range(10):
            detection.redraw_cursor(capture.capture(), True)
