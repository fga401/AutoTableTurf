from tableturf.manager import Closer
from tableturf.manager.data import Stats


class WebCloser(Closer):
    def __init__(self):
        self.__should_close = False

    def set_close(self):
        self.__should_close = True

    def close(self, stats: Stats) -> bool:
        return self.__should_close
