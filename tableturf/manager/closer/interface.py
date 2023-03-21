from abc import ABC

from tableturf.manager.data import JobStats


class Closer(ABC):
    def close(self, job_stats: JobStats) -> bool:
        return False
