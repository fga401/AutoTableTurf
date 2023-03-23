import json
from dataclasses import dataclass
from enum import Enum


class Result(Enum):
    Win = 'win'
    Loss = 'loss'
    Draw = 'draw'


class JobStats:
    def __init__(self):
        self.task_stats = TaskStats()
        self.task_id = 0

    def __repr__(self):
        return f'JobStats(task_id={self.task_id}, task_stats={self.task_stats})'

    def __str__(self):
        return repr(self)


class TaskStats:
    def __init__(self):
        self.win = 0
        self.battle = 0
        self.time = 0
        self.start_time = 0

    def __repr__(self):
        return f'TaskStats(win={self.win}, battle={self.battle}, time={self.time}, start_time={self.start_time})'

    def __str__(self):
        return repr(self)


@dataclass
class Profile:
    @dataclass
    class Task:
        current_level: int
        current_win: int
        target_level: int
        target_win: int
        deck: int

    tasks: list[Task]

    @staticmethod
    def from_json(text: str):
        obj = json.loads(text)
        tasks = [Profile.Task(
            current_level=node['current_level'],
            current_win=node['current_win'],
            target_level=node['target_level'],
            target_win=node['target_win'],
            deck=node['deck'],
        ) for node in obj]
        return Profile(tasks=tasks)
