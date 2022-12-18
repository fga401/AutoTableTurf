from enum import Enum


class Grid(Enum):
    Empty = 1
    MyInk = 2
    MySpecial = 4
    HisInk = 8
    HisSpecial = 16
    Neutral = 32
    Wall = 64
