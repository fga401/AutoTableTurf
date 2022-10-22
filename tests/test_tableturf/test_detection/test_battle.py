import os
import unittest

from capture import FileLoader
from tableturf.manager import detection

path = os.path.join(os.path.realpath(__file__), '..', '..', '..', '..', 'temp', 'stage2')


@unittest.skipIf(not os.path.exists(path), 'images not existed')
class TestBattle(unittest.TestCase):
    def test_hands_cursor(self):
        capture = FileLoader(path=path)
        for _ in range(10):
            detection.hands_cursor(capture.capture(), True)

    def test_hands(self):
        capture = FileLoader(path=path)
        for _ in range(10):
            detection.hands(capture.capture(), None, True)
