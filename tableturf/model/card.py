from typing import Union

import numpy as np

from tableturf.model.grid import Grid


class Card:
    # counterclockwise 90°
    __ROTATION_MATRIX = np.array([
        [0, -1],
        [1, 0],
    ])
    __INVERSE_ROTATION_MATRIX = np.linalg.inv(__ROTATION_MATRIX).astype(np.int)

    def __init__(self, grid: np.ndarray, sp_cost: int):
        """
        Represent a card. Each inked square is assigned an ID, which numbers first from left to right, then from up to down.

        :param grid: Card pattern.
        :param sp_cost: Special Points that a Special Attack costs.
        """
        if isinstance(grid[0][0], Grid):
            self.__grid = np.vectorize(lambda x: x.value)(grid)
        else:
            self.__grid = grid.copy()
        self.__sp_cost = sp_cost
        ss_index = np.argwhere(self.__grid == Grid.MySpecial.value)
        indexes = np.argwhere((self.__grid == Grid.MyInk.value) | (self.__grid == Grid.MySpecial.value))
        self.__ss_id = None
        if ss_index.size > 0:
            self.__ss_id = np.argwhere((indexes == ss_index).all(axis=1))[0]
        self.__offsets = np.array([indexes - idx for idx in indexes])
        self.__size, _ = indexes.shape

        self.__grid.setflags(write=False)

    @property
    def size(self) -> int:
        """
        The number of squares the pattern covers.
        """
        return self.__size

    @property
    def sp_cost(self) -> int:
        """
        Special Points that a Special Attack costs.
        """
        return self.__sp_cost

    @property
    def ss_id(self) -> Union[int, None]:
        """
        Special Space id.
        """
        return self.__ss_id

    def get_grid(self, rotate: int = 0) -> np.ndarray:
        """
        Get the pattern of the card.

        :param rotate: The times of rotation (counterclockwise 90°)
        """
        return np.rot90(self.__grid, rotate)

    def get_offsets(self, origin: int = None, rotate: int = 0) -> np.ndarray:
        """
        Given an origin, return all square offsets of the pattern

        :param origin: The ID of the origin.
        :param rotate: The times of rotation (counterclockwise 90°)
        """
        # return np.matmul(np.linalg.matrix_power(_ROTATION_MATRIX, rotate), self.__offsets[origin_id].T).T
        if origin is None:
            if self.__ss_id is None:
                origin = 0
            else:
                origin = self.__ss_id
        return np.matmul(self.__offsets[origin], np.linalg.matrix_power(Card.__INVERSE_ROTATION_MATRIX, rotate))

    def __hash__(self):
        return hash(str(self.__offsets[0]))

    def __eq__(self, other):
        if isinstance(other, Card):
            return (self.__offsets[0] == other.__offsets[0]).all()
        return NotImplemented

    def __repr__(self):
        return '\ngrid:\n' + str(self.__grid) + '\ncost:' + str(self.__sp_cost) + '\n'
