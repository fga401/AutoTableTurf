from enum import Enum
from typing import Union

import numpy as np

from tableturf.model.card import Card


class Step:
    class Action(Enum):
        Place = 0
        SpecialAttack = 1
        Skip = 2

    def __init__(self, card: Card, rotate: Union[int, None], pos: Union[np.ndarray, None], action: Action):
        self.__card = card
        self.__rotate = rotate
        self.__pos = pos.copy() if pos is not None else None
        self.__action = action

        if pos is not None:
            self.__pos.setflags(write=False)

    @property
    def card(self) -> Card:
        return self.__card

    @property
    def rotate(self) -> Union[int, None]:
        return self.__rotate

    @property
    def pos(self) -> Union[np.ndarray, None]:
        return self.__pos

    @property
    def action(self) -> Action:
        return self.__action

    def __hash__(self):
        return hash((self.card, self.__rotate, self.__action))

    def __eq__(self, other):
        if isinstance(other, Step):
            if self.__pos is None:
                return (self.card, self.__rotate, self.__action) == (other.card, other.__rotate, other.__action) and self.__pos == other.__pos
            else:
                return (self.card, self.__rotate, self.__action) == (other.card, other.__rotate, other.__action) and (self.__pos == other.__pos).all()
        return NotImplemented
