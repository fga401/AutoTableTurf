from typing import List
from urllib.parse import urljoin

import requests

from controller import Controller
from logger import logger


class NxbtController(Controller):
    def __init__(self, endpoint):
        self.__endpoint = endpoint
        self.__press_buttons_endpoint = urljoin(self.__endpoint, 'press_buttons')
        self.__tilt_stick_endpoint = urljoin(self.__endpoint, 'tilt_stick')
        self.__macro_endpoint = urljoin(self.__endpoint, 'macro')

    def press_buttons(self, buttons: List[Controller.Button], down: float = 0.05, up: float = 0.05, block=True) -> bool:
        logger.debug(f'press_buttons: buttons={buttons}, down={down}, up={up}, block={block}')
        buttons = ','.join([str(b.value) for b in buttons])
        params = {'buttons': buttons, 'down': down, 'up': up, 'block': block}
        resp = requests.post(self.__press_buttons_endpoint, params=params)
        return resp.status_code == 200

    def tilt_stick(self, stick: Controller.Stick, x: int, y: int, tilted: float = 0.05, released: float = 0.05, block=True) -> bool:
        logger.debug(f'tilt_stick: stick={stick}, x={x}, y={y}, tilted={tilted}, released={released}, block={block}')
        params = {'stick': str(stick.value), 'x': x, 'y': y, 'tilted': tilted, 'released': released, 'block': block}
        resp = requests.post(self.__tilt_stick_endpoint, params=params)
        return resp.status_code == 200

    def macro(self, macro: str, block=True) -> bool:
        logger.debug(f'macro: macro=\n"""\n{macro.strip()}\n"""\n, block={block}')
        params = {'block': block}
        resp = requests.post(self.__macro_endpoint, params=params, data=macro)
        return resp.status_code == 200
