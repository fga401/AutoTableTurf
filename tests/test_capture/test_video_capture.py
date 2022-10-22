import os
import unittest

from capture import VideoCapture


def ready():
    VideoCapture(0).capture()
    return True


@unittest.skipIf(not ready(), 'nxbt server not ready')
class TestCard(unittest.TestCase):
    def setUp(self) -> None:
        self.capture = VideoCapture(0)

    def test_width(self):
        self.assertEqual(self.capture.width, 1920)

    def test_height(self):
        self.assertEqual(self.capture.height, 1080)

    def test_show(self):
        self.capture.show()

    def test_save(self):
        cur = os.path.dirname(os.path.realpath(__file__))
        root = os.path.join(cur, os.pardir, os.pardir)
        target = os.path.join(root, 'temp', 'stage1', 'p2')
        print(target)
        self.capture.save(target)
