import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import List

from logger import logger


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
    def press_buttons(self, buttons: List[Button], down: float = 0.08, up: float = 0.08, block=True) -> bool:
        raise NotImplementedError

    @abstractmethod
    def tilt_stick(self, stick: Stick, x: int, y: int, tilted: float = 0.1, released: float = 0.1, block=True) -> bool:
        raise NotImplementedError

    # Macro format:
    # https://github.com/Brikwerk/nxbt/blob/master/docs/Macros.md
    def macro(self, macro: str, block=True) -> bool:
        # TODO: validation
        try:
            parsed = [[s.strip() for s in row.split(' ') if s.strip() != ''] for row in macro.split("\n") if row.strip() != '']
            for i, inputs in enumerate(parsed):
                print(inputs)
                if len(inputs) == 1:
                    continue
                down_time = float(inputs[-1][:-1])
                up_time = 0 if i + 1 >= len(parsed) or len(parsed[i + 1]) != 1 else parsed[i + 1][0][:-1]
                buttons = [self.Button(b) for b in inputs[:-1] if not b.startswith('R_STICK@') and not b.startswith('L_STICK@')]
                sticks = [b for b in inputs[:-1] if b.startswith('R_STICK@') or b.startswith('L_STICK@')]
                t_buttons = threading.Thread(target=self.press_buttons, args=(buttons, down_time, up_time, True))
                t_sticks = []
                for stick in sticks:
                    stick, positions = stick.split("@")
                    stick = self.Stick(stick)
                    x = int(positions[0:4])
                    y = int(positions[4:8])
                    t_sticks.append(threading.Thread(target=self.tilt_stick, args=(stick, x, y, down_time, up_time, True)))
                t_buttons.start()
                for t in t_sticks:
                    t.start()
                t_buttons.join()
                for t in t_sticks:
                    t.join()
            # TODO: should fetch result from threads
            return True
        except Exception as e:
            logger.error(f'Controller.macro: exception={e}')
            return False
