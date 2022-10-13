from datetime import datetime
from typing import Union, List

import numpy as np

from controller import Controller
from screen import ScreenCapturer
from tableturf.ai import AI
from tableturf.model import Status, Card, Step


class Stats:
    def __init__(self):
        self.win = 0
        self.battle = 0
        self.time = 0


class Result:
    def __init__(self, my_ink: int, his_ink: int):
        self.my_ink = my_ink
        self.his_ink = his_ink


class TableTureManager:
    class Closer:
        def __init__(self, max_win: Union[int, None] = None, max_battle: Union[int, None] = None, max_time: Union[int, None] = None):
            self.__max_win = max_win
            self.__max_battle = max_battle
            self.__max_time = max_time

        def close(self, stats: Stats) -> bool:
            if self.__max_win is not None and self.__max_win <= stats.win:
                return True
            if self.__max_battle is not None and self.__max_battle <= stats.battle:
                return True
            if self.__max_time is not None and self.__max_time <= stats.time:
                return True
            return False

    def __init__(self, screen_capturer: ScreenCapturer, controller: Controller, ai: AI, closer: Closer = Closer()):
        self.__screen_capturer = screen_capturer
        self.__controller = controller
        self.__ai = ai
        self.__closer = closer
        self.stats = Stats()

    def run(self, my_deck_pos: int, my_deck: Union[List[Card], None] = None, his_deck: Union[List[Card], None] = None):
        # TODO: load my deck from screen
        start_time = datetime.now().timestamp()
        while True:
            self.__select_deck(my_deck_pos)
            screen = self.__screen_capturer.capture()
            status = self.__get_status(screen, my_deck, his_deck)
            redraw = self.__ai.redraw(status)
            self.__redraw(redraw)
            for round in range(15):
                screen = self.__screen_capturer.capture()
                status = self.__get_status(screen, my_deck, his_deck)
                step = self.__ai.next_step(status)
                self.__move(step)
            screen = self.__screen_capturer.capture()
            result = self.__get_result(screen)
            # update stats
            if result.my_ink > result.his_ink:
                self.stats.win += 1
            now = datetime.now().timestamp()
            self.stats.time = now - start_time
            self.stats.battle += 1
            # keep playing
            self.__close(self.__closer.close(self.stats))

    def __get_hands(self, screen: np.ndarray) -> List[Card]:
        pass

    def __get_status(self, screen: np.ndarray, my_deck: Union[List[Card], None] = None, his_deck: Union[List[Card], None] = None) -> Status:
        # TODO: get hands
        # TODO: get stage
        # TODO: get SP
        pass

    def __get_result(self, screen: np.ndarray) -> Result:
        pass

    def __select_deck(self, my_deck_pos: int):
        pass

    def __redraw(self, redraw:bool):
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
