from typing import List

from controller.interface import Controller
from logger import logger


class DummyController(Controller):
    def __init__(self):
        pass

    def press_buttons(self, buttons: List[Controller.Button], down: float = 0.1, up: float = 0.1, block=True) -> bool:
        logger.info(f'press_buttons: buttons={buttons}, down={down}, up={up}, block={block}')
        input()
        return True

    def tilt_stick(self, stick: Controller.Stick, x: int, y: int, tilted: float = 0.1, released: float = 0.1, block=True) -> bool:
        logger.info(f'tilt_stick: stick={stick}, x={x}, y={y}, tilted={tilted}, released={released}, block={block}')
        input()
        return True

    def macro(self, macro: str, block=True) -> bool:
        logger.info(f'macro: macro=\n"""\n{macro.strip()}\n"""\n, block={block}')
        input()
        return True
