from typing import Optional

from tableturf.manager.closer.interface import Closer
from tableturf.manager.data import JobStats


class TaskStatsCloser(Closer):
    def __init__(self, max_win: Optional[int] = None, max_battle: Optional[int] = None, max_time: Optional[int] = None):
        self.__max_win = max_win
        self.__max_battle = max_battle
        self.__max_time = max_time

    def close(self, job_stats: JobStats) -> bool:
        if self.__max_win is not None and self.__max_win <= job_stats.task_stats.win:
            return True
        if self.__max_battle is not None and self.__max_battle <= job_stats.task_stats.battle:
            return True
        if self.__max_time is not None and self.__max_time <= job_stats.task_stats.time:
            return True
        return False
