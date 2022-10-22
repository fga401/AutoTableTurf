import os
import unittest

from capture import FileLoader
from tableturf.manager import detection

path = os.path.join(os.path.realpath(__file__), '..', '..', '..', '..', 'temp', 'deck')


@unittest.skipIf(not os.path.exists(path), 'images not existed')
class TestDeckCursor(unittest.TestCase):
    def test_deck_cursor(self):
        capture = FileLoader(path=path)
        for _ in range(10):
            detection.deck_cursor(capture.capture(), True)
