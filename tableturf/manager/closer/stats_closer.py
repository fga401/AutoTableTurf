from typing import Optional

from tableturf.manager.closer.interface import Closer
from tableturf.manager.data import Stats


class StatsCloser(Closer):
    def __init__(self, max_win: Optional[int] = None, max_battle: Optional[int] = None, max_time: Optional[int] = None):
        self.__max_win = max_win
        self.__max_battle = max_battle
        self.__max_time = max_time

    def close(self, stats: Stats) -> bool:
        if self.__max_win is not None and self.__max_win <= stats.win:
            return True
        if self.__max_battle is not None and self.__max_battle <= stats.battle:
            return True
        if self.__max_time is not None and self.__max_time <= stats.time:
            return True
        return False
