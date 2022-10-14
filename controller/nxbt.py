from typing import List
import requests
from urllib.parse import urljoin

from controller import Controller


class NxbtController(Controller):
    def __init__(self, endpoint):
        self.__endpoint = endpoint
        self.__press_buttons_endpoint = urljoin(self.__endpoint, 'press_buttons')
        self.__tilt_stick_endpoint = urljoin(self.__endpoint, 'tilt_stick')
        self.__macro_endpoint = urljoin(self.__endpoint, 'macro')

    def press_buttons(self, buttons: List[Controller.Button], down: float = 0.1, up: float = 0.1, block=True) -> bool:
        buttons = ','.join([str(b.value) for b in buttons])
        params = {'buttons': buttons, 'down': down, 'up': up, 'block': block}
        resp = requests.post(self.__press_buttons_endpoint, params=params)
        return resp.status_code == 200

    def tilt_stick(self, stick: Controller.Stick, x: int, y: int, tilted: float = 0.1, released: float = 0.1, block=True) -> bool:
        params = {'stick': str(stick.value), 'x': x, 'y': y, 'tilted': tilted, 'released': released, 'block': block}
        resp = requests.post(self.__tilt_stick_endpoint, params=params)
        return resp.status_code == 200

    def macro(self, macro: str, block=True) -> bool:
        params = {'block': block}
        resp = requests.post(self.__macro_endpoint, params=params, data=macro)
        return resp.status_code == 200
