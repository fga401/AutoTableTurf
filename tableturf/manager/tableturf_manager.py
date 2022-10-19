from datetime import datetime
from time import sleep
from typing import Union, List

import cv2
import numpy as np

from capture import Capture
from controller import Controller
from tableturf.ai import AI
from tableturf.manager import action
from tableturf.manager import detection
from tableturf.manager.data import Stats, Result
from tableturf.manager.exit_manager import ExitManager
from tableturf.model import Status, Card, Step


class TableTurfManager:
    @staticmethod
    def resize(capture_fn):
        """
        Resize the captured image to (1920, 1080) to ensure that ROIs work correctly.
        """

        def __resize():
            img = capture_fn()
            height, width, _ = img.shape
            if height != 1080 or width != 1920:
                img = cv2.resize(img, (1920, 1080))
            return img

        return __resize

    @staticmethod
    def multi_detect(detect_fn, times=3):
        def __multi_detect():
            results = [detect_fn() for _ in range(times)]
            return max(set(results), key=results.count)

        return __multi_detect

    @staticmethod
    def feedback(target, detect_fn, action_fn, max_times=5):
        for _ in range(max_times):
            current = detect_fn()
            if current == target:
                break
            action_fn(target, current)
        raise Exception('feedback loop reaches max_times')

    def __init__(self, capture: Capture, controller: Controller, ai: AI, closer: ExitManager = ExitManager(), debug=False):
        self.__capture = self.resize(capture.capture)
        self.__controller = controller
        self.__ai = ai
        self.__closer = closer
        self.__debug = debug
        self.stats = Stats()

    def run(self, deck: int, his_deck: Union[List[Card], None] = None):
        start_time = datetime.now().timestamp()
        while True:
            self.__select_deck(deck)
            my_deck = self.__get_deck()
            self.__confirm_deck()
            sleep(10)  # wait for animate

            # status = self.__get_status(screen, my_deck, his_deck)
            # redraw = self.__ai.redraw(status)
            # self.__redraw(redraw)
            # for round in range(15):
            #     screen = self.__capture()
            #     status = self.__get_status(screen, my_deck, his_deck)
            #     step = self.__ai.next_step(status)
            #     self.__move(step)
            # screen = self.__capture()
            result = self.__get_result()
            # update stats
            if result.my_ink > result.his_ink:
                self.stats.win += 1
            now = datetime.now().timestamp()
            self.stats.time = now - start_time
            self.stats.battle += 1
            # keep playing
            self.__close(self.__closer.exit(self.stats))

    def __select_deck(self, deck: int):
        def get_deck_pos() -> int:
            img = self.__capture()
            return detection.deck_selection.get_deck_pos(img, debug=self.__debug)

        def move_cursor(target: int, current: int):
            macro = action.deck_selection.move_cursor(target, current)
            self.__controller.macro(macro)

        self.feedback(deck, self.multi_detect(get_deck_pos), move_cursor)

    def __get_deck(self) -> List[Card]:
        pass

    def __confirm_deck(self):
        self.__controller.press_buttons([Controller.Button.A])

    def __get_hands(self, screen: np.ndarray) -> List[Card]:
        pass

    def __get_status(self, screen: np.ndarray, my_deck: Union[List[Card], None] = None, his_deck: Union[List[Card], None] = None) -> Status:
        # TODO: get hands
        # TODO: get stage
        # TODO: get SP
        pass

    def __get_result(self) -> Result:
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
