from abc import ABC, abstractmethod

from tableturf.model import Status, Step


class AI(ABC):
    @abstractmethod
    def redraw(self, status: Status) -> bool:
        raise NotImplementedError

    @abstractmethod
    def next_step(self, status: Status) -> Step:
        raise NotImplementedError
