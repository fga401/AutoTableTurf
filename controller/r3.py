import serial
import serial.tools.list_ports

from typing import List
from time import sleep

from controller import Controller
from logger import logger

"""
The following code can use the serial port to control an Arduino UNO R3 that simulates a HORI GamePad.
The following code is improved from https://github.com/wwwwwwzx/Switch-Fightstick

Author: https://github.com/LiuJiLan
"""


class R3Controller(Controller):
    def __init__(self, serial_port=None):
        if serial_port is None:
            serial_port = self.__find_port()
            print(f'Using port: {serial_port[0]}')
        self.__ser = serial.Serial(serial_port[0], 9600)
        self.__chart = dict({
            'Y': 'Button Y',
            'X': 'Button X',
            'B': 'Button B',
            'A': 'Button A',
            'JCL_SR': None,
            'JCL_SL': None,
            'R': 'Button R',
            'ZR': 'Button ZR',
            'MINUS': 'Button MINUS',
            'PLUS': 'Button PLUS',
            'R_STICK_PRESS': 'Button RCLICK',
            'L_STICK_PRESS': 'Button LCLICK',
            'HOME': 'Button HOME',
            'CAPTURE': 'Button CAPTURE',
            'DPAD_DOWN': 'HAT BOTTOM',
            'DPAD_UP': 'HAT TOP',
            'DPAD_RIGHT': 'HAT RIGHT',
            'DPAD_LEFT': 'HAT LEFT',
            'JCR_SR': None,
            'JCR_SL': None,
            'L': 'Button L',
            'ZL': 'Button ZL',
            'R_STICK': 'R',
            'L_STICK': 'L'
        })

    @staticmethod
    def __find_port():
        ports = [
            p.device
            for p in serial.tools.list_ports.comports()
            if p.vid is not None and p.pid is not None
        ]
        if not ports:
            raise IOError('No device found')
        if len(ports) > 1:
            print('Found multiple devices:')
            for p in ports:
                print(p)
        return ports

    def press_buttons(self, buttons: List[Controller.Button], down: float = 0.1, up: float = 0.1, block=True) -> bool:
        logger.debug(f'press_buttons: buttons={buttons}, down={down}, up={up}, block={block}')
        tmp_ls = [self.__chart[str(b.value)] for b in buttons]
        msg = '\r\n'.join([x for x in tmp_ls if x is not None])
        self.__ser.write(f'{msg}\r\n'.encode('utf-8'))
        sleep(down)
        self.__ser.write(b'RELEASE\r\n')
        sleep(up)
        return True

    def tilt_stick(self, stick: Controller.Stick, x: int, y: int, tilted: float = 0.1, released: float = 0.1, block=True) -> bool:
        logger.debug(f'tilt_stick: stick={stick}, x={x}, y={y}, tilted={tilted}, released={released}, block={block}')
        if 128 > x >= 0:
            x_msg = 'MIN'  # 0
        elif 128 < x <= 255:
            x_msg = 'MAX'  # 255
        else:
            x_msg = 'CENTER'

        if 128 > y >= 0:
            y_msg = 'MIN'  # 0
        elif 128 < y <= 255:
            y_msg = 'MAX'  # 255
        else:
            y_msg = 'CENTER'

        stick_msg = self.__chart[str(stick.value)]
        self.__ser.write(f'{stick_msg}X {x_msg}\r\n {stick_msg}Y {y_msg}\r\n'.encode('utf-8'))
        return True
