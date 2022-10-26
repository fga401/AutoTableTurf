from enum import Enum
from typing import Union

import numpy as np

from tableturf.model.card import Card


class Step:
    class Action(Enum):
        Place = 0
        SpecialAttack = 1
        Skip = 2

    def __init__(self, action: Action, card: Card, rotate: Union[int, None], pos: Union[np.ndarray, None]):
        """
        :param pos: (x, y)
        """
        self.__card = card
        self.__rotate = rotate if rotate is not None else None
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
        return hash((self.card, self.__action))

    def __eq__(self, other):
        if isinstance(other, Step):
            self_rotate = 0 if self.__rotate is None else self.__rotate
            other_rotate = 0 if other.__rotate is None else other.__rotate
            return (self.card.get_pattern(self_rotate), self.__action) == (other.card.get_pattern(other_rotate), other.__action) and np.all(self.__pos == other.__pos)
        return NotImplemented

    def __repr__(self):
        return f'Step(action={self.__action}, card={self.__card}, rotate={self.__rotate}, pos={self.__pos})'

    def __str__(self):
        return repr(self)
