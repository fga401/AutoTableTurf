from tableturf.manager.closer.interface import Closer
from tableturf.manager.data import JobStats


class UnionCloser(Closer):
    def __init__(self, this: Closer, that: Closer):
        self.__this = this
        self.__that = that

    def close(self, job_stats: JobStats) -> bool:
        return self.__this.close(job_stats) or self.__that.close(job_stats)
