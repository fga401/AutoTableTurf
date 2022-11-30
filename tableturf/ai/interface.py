from abc import ABC, abstractmethod
from typing import List, Optional

from tableturf.model import Status, Step, Card, Stage


class AI(ABC):
    @abstractmethod
    def redraw(self, hands: List[Card], stage: Optional[Stage] = None, my_whole_deck: Optional[List[Card]] = None, his_whole_deck: Optional[List[Card]] = None) -> bool:
        raise NotImplementedError

    @abstractmethod
    def next_step(self, status: Status) -> Step:
        raise NotImplementedError
