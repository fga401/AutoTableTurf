import os

import cv2
import numpy as np

from capture.interface import Capture


class FileLoader(Capture):
    def __init__(self, *, files=None, path=None):
        """
        Load .npy files as screen capture, which is used for test.

        :param files: List of paths to .npy. If provided, it will use these files as output and ignore the param path.
        :param path: A base path. If provided, it will load all .npy files under this path.
        """
        self.__files = []
        if files is not None:
            self.__files = files
        elif path is not None:
            self.__files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.jpg')]
        self.__idx = 0

    def capture(self) -> np.ndarray:
        frame = cv2.imread(self.__files[self.__idx])
        self.__idx = (self.__idx + 1) % len(self.__files)
        height, width, _ = frame.shape
        if height != 1080 or width != 1920:
            frame = cv2.resize(frame, (1920, 1080))
        return frame

    def close(self):
        return
