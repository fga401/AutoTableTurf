from enum import Enum


class Grid(Enum):
    Empty = 0
    MyInk = 1
    MySpecial = 2
    HisInk = -1
    HisSpecial = -2
    Gray = -10
    Wall = -100
