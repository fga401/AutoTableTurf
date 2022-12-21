import os
import unittest

from capture import FileLoader
from tableturf.manager.detection.debugger import OpenCVDebugger
from tableturf.manager import detection

path = os.path.join(os.path.realpath(__file__), '..', '..', '..', '..', 'temp')


@unittest.skipIf(not os.path.exists(path), 'images not existed')
class TestDeck(unittest.TestCase):
    def test_deck(self):
        capture = FileLoader(path=os.path.join(path, 'deck'))
        for _ in range(10):
            deck = detection.deck(capture.capture(), OpenCVDebugger())
            print(deck)
