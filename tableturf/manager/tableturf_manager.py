from datetime import datetime
from typing import Union, List

import cv2
import numpy as np

from capture import Capture
from controller import Controller
from tableturf.ai import AI
from tableturf.manager import detection
from tableturf.manager.data import Stats, Result
from tableturf.manager.exit_manager import ExitManager
from tableturf.model import Status, Card, Step


class TableTurfManager:
    def __init__(self, capture: Capture, controller: Controller, ai: AI, closer: ExitManager = ExitManager(), debug=False):

        def resize(fn):
            """
            Resize the captured image to (1920, 1080) to ensure that ROIs work correctly.
            """

            def __resize():
                img = fn()
                height, width, _ = img.shape
                if height != 1080 or width != 1920:
                    img = cv2.resize(img, (1920, 1080))
                return img

            return __resize

        self.__capture = resize(capture.capture)
        self.__controller = controller
        self.__ai = ai
        self.__closer = closer
        self.__debug = debug
        self.stats = Stats()

    def run(self, my_deck_pos: int, my_deck: Union[List[Card], None] = None, his_deck: Union[List[Card], None] = None):
        while True:
            img = self.__capture()
            pos = detection.deck_selection.get_deck_pos(img, debug=self.__debug)
            print(pos)
        # TODO: load my deck from capture
        start_time = datetime.now().timestamp()
        while True:
            self.__select_deck(my_deck_pos)
            screen = self.__capture()
            status = self.__get_status(screen, my_deck, his_deck)
            redraw = self.__ai.redraw(status)
            self.__redraw(redraw)
            for round in range(15):
                screen = self.__capture()
                status = self.__get_status(screen, my_deck, his_deck)
                step = self.__ai.next_step(status)
                self.__move(step)
            screen = self.__capture()
            result = self.__get_result(screen)
            # update stats
            if result.my_ink > result.his_ink:
                self.stats.win += 1
            now = datetime.now().timestamp()
            self.stats.time = now - start_time
            self.stats.battle += 1
            # keep playing
            self.__close(self.__closer.exit(self.stats))

    def __select_deck(self, my_deck_pos: int):
        pass

    def __get_hands(self, screen: np.ndarray) -> List[Card]:
        pass

    def __get_status(self, screen: np.ndarray, my_deck: Union[List[Card], None] = None, his_deck: Union[List[Card], None] = None) -> Status:
        # TODO: get hands
        # TODO: get stage
        # TODO: get SP
        pass

    def __get_result(self, screen: np.ndarray) -> Result:
        pass

    def __redraw(self, redraw: bool):
        pass

    def __move(self, step: Step):
        pass

    def __place(self, step: Step):
        pass

    def __special_attack(self, step: Step):
        pass

    def __skip(self, step: Step):
        pass

    def __close(self, close: bool):
        pass
