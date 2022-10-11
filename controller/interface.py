from abc import ABC, abstractmethod
from enum import Enum


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
        RIGHT_STICK = 'R_STICK'
        LEFT_STICK = 'L_STICK'

    @abstractmethod
    def press_buttons(self, buttons: Button, down: float = 0.1, up: float = 0.1):
        raise NotImplementedError

    @abstractmethod
    def tilt_stick(self, stick: Stick, x: float, y: float, tilted: float = 0.1, released: float = 0.1):
        raise NotImplementedError
