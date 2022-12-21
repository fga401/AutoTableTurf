from abc import ABC

from tableturf.manager.data import Stats


class Closer(ABC):
    def close(self, stats: Stats) -> bool:
        raise NotImplementedError
