import datetime

from tableturf.manager import Closer
from tableturf.manager.data import JobStats


class WebCloser(Closer):
    def __init__(self, stop_at: datetime.datetime = None):
        self.__should_close = False
        self.__stop_at = stop_at

    def set_close(self):
        self.__should_close = True

    def close(self, job_stats: JobStats) -> bool:
        if self.__stop_at is not None and datetime.datetime.now() > self.__stop_at:
            return True
        return self.__should_close
