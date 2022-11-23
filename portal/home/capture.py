import numpy as np

from capture import Capture
from portal.util.rwlock import RWLock


class ThreadSafeCapture(Capture):
    def __init__(self, capture: Capture):
        self.lock = RWLock()
        self.__capture = capture

    def capture(self) -> np.ndarray:
        with self.lock.r_locked():
            return self.__capture.capture()

    def update_capture(self, capture: Capture):
        with self.lock.w_locked():
            self.__capture.close()
            self.__capture = capture

    def close(self):
        with self.lock.w_locked():
            self.__capture.close()
