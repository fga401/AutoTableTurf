from typing import List, Set, Optional

import numpy as np

from tableturf.model.card import Card
from tableturf.model.grid import Grid
from tableturf.model.stage import Stage
from tableturf.model.step import Step


class Status:
    def __init__(self, stage: Stage, hands: List[Card], my_sp: int, his_sp: int, my_deck: List[Card], his_deck: List[Card]):
        self.__stage = stage
        self.__hands = hands
        self.__my_sp = my_sp
        self.__his_sp = his_sp
        self.__my_deck = my_deck
        self.__his_deck = his_deck

        m, n = self.stage.shape
        neighborhoods = self.stage.my_neighborhoods
        sp_neighborhoods = self.stage.my_sp_neighborhoods

        def possible_steps_without_special_attack(card: Card) -> Set[Step]:
            result = set()
            result.add(Step(Step.Action.Skip, card, None, None))
            for idx in neighborhoods:
                for origin in range(card.size):
                    for rotate in range(4):
                        pattern_indexes = card.get_pattern(rotate).offset + idx
                        # pattern is out of boundary
                        xs = pattern_indexes[:, 0]
                        ys = pattern_indexes[:, 1]
                        if not np.all((xs >= 0) & (xs < m) & (ys >= 0) & (ys < n)):
                            continue
                        # squares are not empty
                        if not np.all(self.stage.grid[xs, ys] == Grid.Empty.value):
                            continue
                        result.add(Step(Step.Action.Place, card, rotate, idx))
            return result

        def possible_steps_with_special_attack(card: Card) -> Set[Step]:
            result = set()
            if card.sp_cost > self.__my_sp:
                return result
            for idx in sp_neighborhoods:
                for origin in range(card.size):
                    for rotate in range(4):
                        pattern_indexes = card.get_pattern(rotate).offset + idx
                        xs = pattern_indexes[:, 0]
                        ys = pattern_indexes[:, 1]
                        # pattern is out of boundary
                        if not np.all((xs >= 0) & (xs < m) & (ys >= 0) & (ys < n)):
                            continue
                        # squares are not empty
                        values = self.stage.grid[xs, ys]
                        if not np.all((values == Grid.Empty.value) | (values == Grid.MyInk.value) | (values == Grid.HisInk.value)):
                            continue
                        result.add(Step(Step.Action.SpecialAttack, card, rotate, idx))
            return result

        self.__all_possible_steps_by_card = {
            card: list(possible_steps_without_special_attack(card).union(possible_steps_with_special_attack(card))) for card in hands
        }
        self.__all_possible_steps = [step for steps_set in self.__all_possible_steps_by_card.values() for step in steps_set]

    @property
    def stage(self) -> Stage:
        return self.__stage

    @property
    def hands(self) -> List[Card]:
        return self.__hands

    @property
    def my_sp(self) -> int:
        return self.__my_sp

    @property
    def his_sp(self) -> int:
        return self.__his_sp

    @property
    def my_deck(self) -> List[Card]:
        return self.__my_deck

    @property
    def his_deck(self) -> List[Card]:
        return self.__his_deck

    def get_possible_steps(self, card: Optional[Card] = None) -> List[Step]:
        if card is None:
            return self.__all_possible_steps
        return self.__all_possible_steps_by_card[card]

    def __repr__(self):
        return f'Stage(stage={self.__stage}, hands={self.__hands}, my_sp={self.__my_sp}, his_sp={self.__his_sp}, my_deck={self.__my_deck}, his_deck={self.__his_deck})'

    def __str__(self):
        return repr(self)
