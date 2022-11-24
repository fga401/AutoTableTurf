from datetime import datetime
from time import sleep
from typing import List, Optional

import cv2

from capture import Capture
from controller import Controller
from tableturf.ai import AI
from tableturf.debugger.interface import Debugger
from tableturf.manager import action
from tableturf.manager import detection
from tableturf.manager.data import Stats, Result
from tableturf.model import Status, Card, Step, Stage


class Exit:
    def __init__(self, max_win: Optional[int] = None, max_battle: Optional[int] = None, max_time: Optional[int] = None):
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

    def __init__(self, capture: Capture, controller: Controller, ai: AI, closer: Exit = Exit(), debug: Optional[Debugger] = None):
        self.__capture = self.__resize(capture.capture)
        self.__controller = controller
        self.__ai = ai
        self.__closer = closer
        self.__debug = debug
        self.stats = Stats()
        self.__session = dict()

    def run(self, deck: int, his_deck: Optional[List[Card]] = None):
        self.stats.start_time = datetime.now().timestamp()
        while True:
            sleep(5)
            my_deck = self.__select_deck(deck)
            sleep(10)
            self.__redraw(my_deck=my_deck)
            sleep(5)
            self.__init_roi()
            for round in range(12):
                status = self.__get_status(my_deck, his_deck)
                step = self.__ai.next_step(status)
                self.__move(status, step)
                sleep(10)
            sleep(5)
            result = self.__get_result()
            self.__update_stats(result)
            self.__close(self.__closer.exit(self.stats))

    def __select_deck(self, deck: int) -> List[Card]:
        target = deck
        current = detection.deck_cursor(self.__capture(), debug=self.__debug)
        macro = action.move_deck_cursor_marco(target, current)
        if macro.strip() != '':
            self.__controller.macro(macro)
        self.__controller.press_buttons([Controller.Button.A])
        return None

    def __redraw(self, stage: Optional[Stage] = None, my_deck: Optional[List[Card]] = None, his_deck: Optional[List[Card]] = None):
        hands = detection.hands(self.__capture(), debug=self.__debug)
        redraw = self.__ai.redraw(hands, stage, my_deck, his_deck)
        target = 1 if redraw else 0
        current = detection.redraw_cursor(self.__capture(), debug=self.__debug)
        macro = action.move_redraw_cursor_marco(target, current)
        if macro.strip() != '':
            self.__controller.macro(macro)
        self.__controller.press_buttons([Controller.Button.A])

    def __init_roi(self):
        img = self.__capture()
        rois, roi_width, roi_height = detection.stage_rois(img, debug=self.__debug)
        self.__session['rois'] = rois
        self.__session['roi_width'] = roi_width
        self.__session['roi_height'] = roi_height
        self.__session['last_stage'] = None
        self.__session['last_is_fiery'] = None

    def __get_status(self, my_deck: Optional[List[Card]] = None, his_deck: Optional[List[Card]] = None) -> Status:
        img = self.__capture()
        rois, roi_width, roi_height, last_stage = self.__session['rois'], self.__session['roi_width'], self.__session['roi_height'], self.__session['last_stage']
        stage, is_fiery = detection.stage(img, rois, roi_width, roi_height, last_stage, debug=self.__debug)
        self.__session['last_stage'], self.__session['last_is_fiery'] = stage, is_fiery
        hands = detection.hands(img, debug=self.__debug)
        my_sp, his_sp = detection.sp(img, debug=self.__debug)
        return Status(stage, hands, my_sp, his_sp, my_deck, his_deck)

    def __move_hands_cursor(self, target):
        current = detection.hands_cursor(self.__capture(), debug=self.__debug)
        macro = action.move_hands_cursor_marco(target, current)
        if macro.strip() != '':
            self.__controller.macro(macro)

    def __move(self, status: Status, step: Step):
        if step.Action == step.Action.Skip:
            self.__move_hands_cursor(5)
            self.__controller.press_buttons([Controller.Button.A])
            self.__move_hands_cursor(status.hands.index(step.card))
            self.__controller.press_buttons([Controller.Button.A])
            return

        if step.Action == step.Action.SpecialAttack:
            self.__move_hands_cursor(5)
            self.__controller.press_buttons([Controller.Button.A])
        self.__move_hands_cursor(status.hands.index(step.card))
        self.__controller.press_buttons([Controller.Button.A])
        if step.rotate > 0:
            macro = action.rotate_card_marco(step)
            if macro.strip() != '':
                self.__controller.macro(macro)
        preview, current_index = detection.preview(self.__capture(), status.stage, self.__session['last_is_fiery'], self.__session['rois'], self.__session['roi_width'], self.__session['roi_height'], self.__debug)
        macro = action.move_card_marco(current_index, preview, status.stage, step)
        if macro.strip() != '':
            self.__controller.macro(macro)
        self.__controller.press_buttons([Controller.Button.A])

    def __get_result(self) -> Result:
        # TODO
        img = self.__capture()
        return Result(0, 0)

    def __update_stats(self, result: Result):
        if result.my_ink > result.his_ink:
            self.stats.win += 1
        now = datetime.now().timestamp()
        self.stats.time = now - self.stats.start_time
        self.stats.battle += 1

    def __close(self, close: bool):
        target = 1 if close else 0
        current = detection.replay_cursor(self.__capture(), debug=self.__debug)
        macro = action.move_replay_cursor_marco(target, current)
        if macro.strip() != '':
            self.__controller.macro(macro)
        self.__controller.press_buttons([Controller.Button.A])
