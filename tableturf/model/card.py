import numpy as np

from tableturf.model.grid import Grid


class Card:
    # counterclockwise 90°
    __ROTATION_MATRIX = np.array([
        [0, -1],
        [1, 0],
    ])
    __INVERSE_ROTATION_MATRIX = np.linalg.inv(__ROTATION_MATRIX).astype(np.int)

    def __init__(self, id_: int, grid: np.ndarray, sp_cost: int):
        """
        Represent a card. Each inked square is assigned an ID, which numbers first from left to right, then from up to down.

        :param id_: Card ID
        :param grid: Card pattern.
        :param sp_cost: Special Points that a Special Attack costs.
        """
        self.__id = id_
        self.__grid = grid.copy()
        self.__sp_cost = sp_cost

        def diff(idx: np.ndarray) -> np.ndarray:
            return indexes - idx

        ss_index = np.argwhere(self.__grid == Grid.MySpecial.value)
        indexes = np.argwhere((self.__grid == Grid.MyInk.value) | (self.__grid == Grid.MySpecial.value))
        self.__ss_id = np.argwhere((indexes == ss_index).all(axis=1))[0]
        self.__offsets = np.apply_along_axis(diff, axis=1, arr=indexes)
        self.__size, _ = indexes.shape

        self.__grid.setflags(write=False)

    @property
    def id(self) -> int:
        """
        Card id.
        """
        return self.__id

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
    def ss_id(self) -> int:
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
            origin = self.__ss_id
        return np.matmul(self.__offsets[origin], np.linalg.matrix_power(Card.__INVERSE_ROTATION_MATRIX, rotate))

    def __hash__(self):
        return hash(self.__id)

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.__id == other.__id
        return NotImplemented
