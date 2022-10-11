from typing import Union

from tableturf.model.card import Card
from tableturf.model.grid import Grid
from tableturf.model.stage import Stage
from tableturf.model.step import Step


class Status:
    def __init__(self, stage: Stage, hands: list[Card], my_sp: int, his_sp: int, my_deck: list[Card], his_deck: list[Card]):
        self.__stage = stage
        self.__hands = hands
        self.__my_sp = my_sp
        self.__his_sp = his_sp
        self.__my_deck = my_deck
        self.__his_deck = his_deck

        m, n = self.stage.shape
        neighborhoods = self.stage.my_neighborhoods
        sp_neighborhoods = self.stage.my_sp_neighborhoods

        def possible_steps_without_special_attack(card: Card) -> set[Step]:
            result = set()
            result.add(Step(card, None, None, Step.Action.Skip))
            for idx in neighborhoods:
                for origin in range(card.size):
                    for rotate in range(4):
                        pattern_indexes = card.get_offsets(origin, rotate) + idx
                        # pattern is out of boundary
                        xs = pattern_indexes[:, 0]
                        ys = pattern_indexes[:, 1]
                        if not ((xs >= 0) & (xs < m) & (ys >= 0) & (ys < n)).all():
                            continue
                        # squares are not empty
                        if not (self.stage.grid[xs, ys] == Grid.Empty.value).all():
                            continue
                        pos = pattern_indexes[card.ss_id]
                        result.add(Step(card, rotate, pos, Step.Action.Place))
            return result

        def possible_steps_with_special_attack(card: Card) -> set[Step]:
            result = set()
            if card.sp_cost > self.__my_sp:
                return result
            for idx in sp_neighborhoods:
                for origin in range(card.size):
                    for rotate in range(4):
                        pattern_indexes = card.get_offsets(origin, rotate) + idx
                        xs = pattern_indexes[:, 0]
                        ys = pattern_indexes[:, 1]
                        # pattern is out of boundary
                        if not ((xs >= 0) & (xs < m) & (ys >= 0) & (ys < n)).all():
                            continue
                        # squares are not empty
                        values = self.stage.grid[xs, ys]
                        if not ((values == Grid.Empty.value) | (values == Grid.MyInk.value) | (values == Grid.HisInk.value)).all():
                            continue
                        pos = pattern_indexes[card.ss_id]
                        result.add(Step(card, rotate, pos, Step.Action.SpecialAttack))
            return result

        self.__all_possible_steps_by_card = {
            card.id: possible_steps_without_special_attack(card).union(possible_steps_with_special_attack(card)) for card in hands
        }
        self.__all_possible_steps = {step for steps_set in self.__all_possible_steps_by_card.values() for step in steps_set}

    @property
    def stage(self) -> Stage:
        return self.__stage

    @property
    def hands(self) -> list[Card]:
        return self.__hands

    @property
    def my_sp(self) -> int:
        return self.__my_sp

    @property
    def his_sp(self) -> int:
        return self.__his_sp

    @property
    def my_deck(self) -> list[Card]:
        return self.__my_deck

    @property
    def his_deck(self) -> list[Card]:
        return self.__his_deck

    def get_possible_steps(self, card_id: Union[int, None] = None) -> set[Step]:
        if card_id is None:
            return self.__all_possible_steps
        return self.__all_possible_steps_by_card[card_id]
