import io

import cv2
import numpy as np

from tableturf.debugger.interface import Debugger


class WebDebugger(Debugger):
    def __init__(self):
        __empty = np.zeros((1080, 1920, 3))
        _, __empty_buf = cv2.imencode(".jpeg", __empty)
        self.__empty_buf = io.BytesIO(__empty_buf)
        # no lock here as Python has GIL and most dict operations are thread-safe
        self.__buffers = dict()

    def show(self, name: str, img: np.ndarray):
        _, buffer = cv2.imencode(".jpeg", img)
        buf = io.BytesIO(buffer)
        self.__buffers[name] = buf

    def get(self, name: str) -> io.BytesIO:
        return self.__buffers.setdefault(name, self.__empty_buf)

    def list(self):
        return self.__buffers.keys()


web_debugger = WebDebugger()
