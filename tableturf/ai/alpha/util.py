import copy
from typing import List

import numpy as np

from tableturf.model import Card, Stage, Step, Status


class RoundConfig:
    def __init__(self, _12, _11, _10, _09, _08, _07, _06, _05, _04, _03, _02, _01):
        self._12 = _12
        self._11 = _11
        self._10 = _10
        self._09 = _09
        self._08 = _08
        self._07 = _07
        self._06 = _06
        self._05 = _05
        self._04 = _04
        self._03 = _03
        self._02 = _02
        self._01 = _01

    def get(self, idx: int):
        if idx == 12:
            return self._12
        elif idx == 11:
            return self._11
        elif idx == 10:
            return self._10
        elif idx == 9:
            return self._09
        elif idx == 8:
            return self._08
        elif idx == 7:
            return self._07
        elif idx == 6:
            return self._06
        elif idx == 5:
            return self._05
        elif idx == 4:
            return self._04
        elif idx == 3:
            return self._03
        elif idx == 2:
            return self._02
        elif idx == 1:
            return self._01
        else:
            return None

    def __getitem__(self, idx: int):
        return self.get(idx)


def is_special_card(card: Card) -> bool:
    return card.size == 12 and card.sp_cost == 3


def has_special_card(cards: List[Card]):
    return next((card for card in cards if is_special_card(card)), None) is not None


def estimate_stage(stage: Stage, step: Step) -> Stage:
    if step.action == Step.Action.Skip:
        return stage
    grid = stage.grid.copy()
    pattern = step.card.get_pattern(step.rotate)
    offset = pattern.offset + step.pos[np.newaxis, ...]
    grid[offset[:, 0], offset[:, 1]] = pattern.squares
    return Stage(grid)


def estimate_my_sp_diff(current_stage: Stage, next_stage: Stage, step: Step) -> int:
    if step.action == Step.Action.Skip:
        return 1
    elif step.action == Step.Action.SpecialAttack:
        return len(next_stage.my_fiery_sp) - len(current_stage.my_fiery_sp) - step.card.sp_cost
    else:  # step.action == Step.Action.Place:
        return len(next_stage.my_fiery_sp) - len(current_stage.my_fiery_sp)


def estimate_his_sp_diff(current_stage: Stage, next_stage: Stage, step: Step):
    if step.action == Step.Action.Skip:
        return 0
    else:
        return len(next_stage.his_fiery_sp) - len(current_stage.his_fiery_sp)


def estimate_status(status: Status, step: Step, expand: bool) -> List[Status]:
    hands = copy.deepcopy(status.hands)
    hands.remove(step.card)
    next_stage = estimate_stage(status.stage, step)
    my_sp = status.my_sp + estimate_my_sp_diff(status.stage, next_stage, step)

    if not expand or len(status.my_deck) == 0:
        return [Status(stage=next_stage, hands=hands, round=status.round - 1, my_sp=my_sp, his_sp=status.his_sp, my_deck=status.my_deck, his_deck=status.his_deck)]
    else:
        result = []
        for card in status.my_deck:
            _hands = hands + [card]
            _my_deck = copy.deepcopy(status.my_deck)
            _my_deck.remove(card)
            result.append(Status(stage=next_stage, hands=_hands, round=status.round - 1, my_sp=my_sp, his_sp=status.his_sp, my_deck=_my_deck, his_deck=status.his_deck))
        return result


def min_max_normalization(arr: np.ndarray) -> np.ndarray:
    min_val = np.min(arr)
    max_val = np.max(arr)
    diff = max_val - min_val
    if diff == 0:
        return np.ones_like(arr)
    return (arr - min_val) / diff


def pause(fn):
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        input()
        return result

    return wrapper
