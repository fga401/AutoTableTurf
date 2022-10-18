from abc import ABC
from typing import Union

from tableturf.manager.data import Stats


class ExitManager(ABC):
    def __init__(self, max_win: Union[int, None] = None, max_battle: Union[int, None] = None, max_time: Union[int, None] = None):
        self.__max_win = max_win
        self.__max_battle = max_battle
        self.__max_time = max_time

    def exit(self, stats: Stats) -> bool:
        if self.__max_win is not None and self.__max_win <= stats.win:
            return True
        if self.__max_battle is not None and self.__max_battle <= stats.battle:
            return True
        if self.__max_time is not None and self.__max_time <= stats.time:
            return True
        return False
