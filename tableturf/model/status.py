from typing import List, Set, Optional

import numpy as np

from tableturf.model.card import Card
from tableturf.model.grid import Grid
from tableturf.model.stage import Stage
from tableturf.model.step import Step


class Status:
    def __init__(self, stage: Stage, hands: List[Card], round: int, my_sp: int, his_sp: int, my_deck: List[Card], his_deck: List[Card]):
        self.__stage = stage
        self.__hands = hands
        self.__round = round
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
                for rotate in range(4):
                    offset = card.get_pattern(rotate).offset
                    for origin in range(card.size):
                        pattern_indexes = offset - offset[origin] + idx
                        # pattern is out of boundary
                        xs = pattern_indexes[:, 0]
                        ys = pattern_indexes[:, 1]
                        if not np.all((xs >= 0) & (xs < m) & (ys >= 0) & (ys < n)):
                            continue
                        # squares are not empty
                        if not np.all(self.stage.grid[xs, ys] == Grid.Empty.value):
                            continue
                        result.add(Step(Step.Action.Place, card, rotate, pattern_indexes[0]))
            return result

        def possible_steps_with_special_attack(card: Card) -> Set[Step]:
            result = set()
            if card.sp_cost > self.__my_sp:
                return result
            for idx in sp_neighborhoods:
                for rotate in range(4):
                    offset = card.get_pattern(rotate).offset
                    for origin in range(card.size):
                        pattern_indexes = offset - offset[origin] + idx
                        xs = pattern_indexes[:, 0]
                        ys = pattern_indexes[:, 1]
                        # pattern is out of boundary
                        if not np.all((xs >= 0) & (xs < m) & (ys >= 0) & (ys < n)):
                            continue
                        # squares are not empty
                        values = self.stage.grid[xs, ys]
                        if not np.all((values == Grid.Empty.value) | (values == Grid.MyInk.value) | (values == Grid.HisInk.value)):
                            continue
                        result.add(Step(Step.Action.SpecialAttack, card, rotate, pattern_indexes[0]))
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
    def round(self) -> int:
        return self.__round

    @property
    def my_sp(self) -> int:
        return self.__my_sp

    @property
    def his_sp(self) -> int:
        return self.__his_sp

    @property
    def my_deck(self) -> List[Card]:
        """
        Return remaining cards in my deck.
        """
        return self.__my_deck

    @property
    def his_deck(self) -> List[Card]:
        """
        Return remaining cards in opponent's deck.
        """
        return self.__his_deck

    def get_possible_steps(self, card: Optional[Card] = None) -> List[Step]:
        """
        Return all possible steps in the current status.

        :param card: if not None, return possible steps of the given card.
        """
        if card is None:
            return self.__all_possible_steps
        return self.__all_possible_steps_by_card[card]

    def __repr__(self):
        return f'Stage(stage={self.__stage}, hands={self.__hands}, my_sp={self.__my_sp}, his_sp={self.__his_sp}, my_deck={self.__my_deck}, his_deck={self.__his_deck})'

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        if isinstance(other, Status):
            return self.__stage == other.__stage and self.__hands == other.__hands and self.__my_sp == other.__my_sp and self.__his_sp == other.__his_sp and self.__my_deck == other.__my_deck and self.__his_deck == other.__his_deck
        return False
