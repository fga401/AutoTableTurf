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
from tableturf.model import Status, Card, Step


class Exit:
    def __init__(self, max_win: Union[int, None] = None, max_battle: Union[int, None] = None, max_time: Union[int, None] = None):
        self.__max_win = max_win
        self.__max_battle = max_battle
        self.__max_time = max_time

    def exit(self, stats: Stats) -> bool:
        if self.__max_win is not None and self.__max_win <= stats.win:
            return True
        if self.__max_battle is not None and self.__max_battle <= stats.battle:
            return True
        if self.__max_time is not None and self.__max_time <= stats.time:
            return True
        return False


class TableTurfManager:
    @staticmethod
    def __resize(capture_fn):
        """
        Resize the captured image to (1920, 1080) to ensure that ROIs work correctly.
        """

        def wrapper():
            img = capture_fn()
            height, width, _ = img.shape
            if height != 1080 or width != 1920:
                img = cv2.resize(img, (1920, 1080))
            return img

        return wrapper

    @staticmethod
    def __multi_detect(detect_fn, times=3):
        def wrapper():
            results = [detect_fn() for _ in range(times)]
            return max(set(results), key=results.count)

        return wrapper

    @staticmethod
    def __feedback(target, detect_fn, action_fn, max_times=5):
        for _ in range(max_times):
            current = detect_fn()
            if current == target:
                return
            action_fn(target, current)
        raise Exception('feedback loop reaches max_times')

    def __init__(self, capture: Capture, controller: Controller, ai: AI, closer: Exit = Exit(), debug=False):
        self.__capture = self.__resize(capture.capture)
        self.__controller = controller
        self.__ai = ai
        self.__closer = closer
        self.__debug = debug
        self.stats = Stats()

    def run(self, deck: int, his_deck: Union[List[Card], None] = None):
        # test begin
        while True:
            self.__redraw()
        # test end
        start_time = datetime.now().timestamp()
        while True:
            self.__select_deck(deck)
            my_deck = self.__get_deck()
            sleep(10)  # wait for animate
            self.__redraw()
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
        def deck_cursor() -> int:
            img = self.__capture()
            return detection.deck_cursor(img, debug=self.__debug)

        def move_cursor(target: int, current: int):
            macro = action.move_deck_cursor_marco(target, current)
            self.__controller.macro(macro)

        self.__feedback(deck, self.__multi_detect(deck_cursor), move_cursor)
        self.__controller.press_buttons([Controller.Button.A])

    def __get_deck(self) -> List[Card]:
        # TODO
        pass

    def __redraw(self):
        status = None  # TODO
        redraw = self.__ai.redraw(status)
        target = 1 if redraw else 0

        def redraw_cursor() -> int:
            img = self.__capture()
            return detection.redraw_cursor(img, debug=self.__debug)

        def move_cursor(target: int, current: int):
            macro = action.move_redraw_cursor_marco(target, current)
            self.__controller.macro(macro)

        self.__feedback(target, self.__multi_detect(redraw_cursor), move_cursor)
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
