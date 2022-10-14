from abc import ABC, abstractmethod
from enum import Enum

from typing import List


class Controller(ABC):
    class Button(Enum):
        Y = 'Y'
        X = 'X'
        B = 'B'
        A = 'A'
        JCL_SR = 'JCL_SR'
        JCL_SL = 'JCL_SL'
        R = 'R'
        ZR = 'ZR'
        MINUS = 'MINUS'
        PLUS = 'PLUS'
        R_STICK_PRESS = 'R_STICK_PRESS'
        L_STICK_PRESS = 'L_STICK_PRESS'
        HOME = 'HOME'
        CAPTURE = 'CAPTURE'
        DPAD_DOWN = 'DPAD_DOWN'
        DPAD_UP = 'DPAD_UP'
        DPAD_RIGHT = 'DPAD_RIGHT'
        DPAD_LEFT = 'DPAD_LEFT'
        JCR_SR = 'JCR_SR'
        JCR_SL = 'JCR_SL'
        L = 'L'
        ZL = 'ZL'

    class Stick(Enum):
        R_STICK = 'R_STICK'
        L_STICK = 'L_STICK'

    @abstractmethod
    def press_buttons(self, buttons: List[Button], down: float = 0.1, up: float = 0.1, block=True) -> bool:
        raise NotImplementedError

    @abstractmethod
    def tilt_stick(self, stick: Stick, x: int, y: int, tilted: float = 0.1, released: float = 0.1, block=True) -> bool:
        raise NotImplementedError

    @abstractmethod
    # Macro format:
    # https://github.com/Brikwerk/nxbt/blob/master/docs/Macros.md
    def macro(self, macro: str, block=True) -> bool:
        raise NotImplementedError
