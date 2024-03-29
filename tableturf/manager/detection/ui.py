from typing import Optional

import numpy as np

from logger import logger
from tableturf.manager.data import Result
from tableturf.manager.detection import util
from tableturf.manager.detection.debugger import Debugger

DECK_CURSOR_ROI_TOP_LEFT = np.array([285, 450])
DECK_CURSOR_ROI_TOP = 285
DECK_CURSOR_ROI_WIDTH_STEP = 345
DECK_CURSOR_ROI_HEIGHT_STEP = 95
DECK_CURSOR_ROI_WIDTH = 15
DECK_CURSOR_ROI_HEIGHT = 15
DECK_CURSOR_NUMPY_ROI_TOP_LEFTS = util.grid_roi_top_lefts(DECK_CURSOR_ROI_TOP_LEFT, 2, 8, DECK_CURSOR_ROI_WIDTH_STEP, DECK_CURSOR_ROI_HEIGHT_STEP, 0, 0).transpose((1, 0, 2)).reshape((16, 2))
DECK_CURSOR_COLOR_HSV_UPPER_BOUND = (35, 255, 255)
DECK_CURSOR_COLOR_HSV_LOWER_BOUND = (30, 100, 150)
DECK_CURSOR_PIXEL_RATIO = 0.2


def deck_cursor(img, debug: Optional[Debugger] = None) -> int:
    pos = util.detect_cursor(
        img,
        DECK_CURSOR_NUMPY_ROI_TOP_LEFTS,
        DECK_CURSOR_ROI_WIDTH,
        DECK_CURSOR_ROI_HEIGHT,
        [(DECK_CURSOR_COLOR_HSV_LOWER_BOUND, DECK_CURSOR_COLOR_HSV_UPPER_BOUND)],
        DECK_CURSOR_PIXEL_RATIO,
        debug,
        'deck_cursor'
    )
    logger.debug(f'detection.deck_cursor: return={pos}')
    return pos


HANDS_CURSOR_NUMPY_ROI_TOP_LEFTS = np.array([[470, 14], [480, 266], [793, 14], [797, 266], [909, 14], [898, 266]])
HANDS_CURSOR_ROI_WIDTH = 30
HANDS_CURSOR_ROI_HEIGHT = 30
HANDS_CURSOR_COLOR_HSV_UPPER_BOUND = (35, 255, 255)
HANDS_CURSOR_COLOR_HSV_LOWER_BOUND = (30, 100, 150)
HANDS_CURSOR_PIXEL_RATIO = 0.3


def hands_cursor(img: np.ndarray, debug: Optional[Debugger] = None) -> int:
    pos = util.detect_cursor(
        img,
        HANDS_CURSOR_NUMPY_ROI_TOP_LEFTS,
        HANDS_CURSOR_ROI_WIDTH,
        HANDS_CURSOR_ROI_HEIGHT,
        [(HANDS_CURSOR_COLOR_HSV_LOWER_BOUND, HANDS_CURSOR_COLOR_HSV_UPPER_BOUND)],
        HANDS_CURSOR_PIXEL_RATIO,
        debug,
        'hands_cursor'
    )
    logger.debug(f'detection.hands_cursor: return={pos}')
    return pos


REDRAW_CURSOR_NUMPY_ROI_TOP_LEFTS = np.array([[760, 900], [760, 1250]])
REDRAW_CURSOR_ROI_WIDTH = 180
REDRAW_CURSOR_ROI_HEIGHT = 70
REDRAW_CURSOR_COLOR_HSV_UPPER_BOUND = (50, 255, 255)
REDRAW_CURSOR_COLOR_HSV_LOWER_BOUND = (30, 100, 150)
REDRAW_CURSOR_PIXEL_RATIO = 0.5


def redraw_cursor(img: np.ndarray, debug: Optional[Debugger] = None) -> int:
    pos = util.detect_cursor(
        img,
        REDRAW_CURSOR_NUMPY_ROI_TOP_LEFTS,
        REDRAW_CURSOR_ROI_WIDTH,
        REDRAW_CURSOR_ROI_HEIGHT,
        [(REDRAW_CURSOR_COLOR_HSV_LOWER_BOUND, REDRAW_CURSOR_COLOR_HSV_UPPER_BOUND)],
        REDRAW_CURSOR_PIXEL_RATIO,
        debug,
        'redraw_cursor'
    )
    logger.debug(f'detection.redraw_cursor: return={pos}')
    return pos


SPECIAL_ON_CURSOR_NUMPY_ROI_TOP_LEFTS = np.array([[840, 295]])
SPECIAL_ON_CURSOR_ROI_WIDTH = 200
SPECIAL_ON_CURSOR_ROI_HEIGHT = 50
SPECIAL_ON_CURSOR_COLOR_HSV_UPPER_BOUND = (40, 255, 240)
SPECIAL_ON_CURSOR_COLOR_HSV_LOWER_BOUND = (20, 150, 200)
SPECIAL_ON_CURSOR_DARK_COLOR_HSV_UPPER_BOUND = (115, 50, 150)
SPECIAL_ON_CURSOR_DARK_COLOR_HSV_LOWER_BOUND = (70, 0, 100)
SPECIAL_ON_CURSOR_PIXEL_RATIO = 0.4


def special_on(img: np.ndarray, debug: Optional[Debugger] = None) -> bool:
    result = util.detect_cursor(
        img,
        SPECIAL_ON_CURSOR_NUMPY_ROI_TOP_LEFTS,
        SPECIAL_ON_CURSOR_ROI_WIDTH,
        SPECIAL_ON_CURSOR_ROI_HEIGHT,
        [(SPECIAL_ON_CURSOR_COLOR_HSV_LOWER_BOUND, SPECIAL_ON_CURSOR_COLOR_HSV_UPPER_BOUND), (SPECIAL_ON_CURSOR_DARK_COLOR_HSV_LOWER_BOUND, SPECIAL_ON_CURSOR_DARK_COLOR_HSV_UPPER_BOUND)],
        SPECIAL_ON_CURSOR_PIXEL_RATIO,
        debug,
        'special_on'
    ) != -1
    logger.debug(f'detection.special_on: return={result}')
    return result


SKIP_CURSOR_NUMPY_ROI_TOP_LEFTS = np.array([[850, 45]])
SKIP_CURSOR_ROI_WIDTH = 200
SKIP_CURSOR_ROI_HEIGHT = 50
SKIP_CURSOR_COLOR_HSV_UPPER_BOUND = (40, 255, 240)
SKIP_CURSOR_COLOR_HSV_LOWER_BOUND = (20, 150, 200)
SKIP_CURSOR_PIXEL_RATIO = 0.4


def skip(img: np.ndarray, debug: Optional[Debugger] = None) -> bool:
    result = util.detect_cursor(
        img,
        SKIP_CURSOR_NUMPY_ROI_TOP_LEFTS,
        SKIP_CURSOR_ROI_WIDTH,
        SKIP_CURSOR_ROI_HEIGHT,
        [(SKIP_CURSOR_COLOR_HSV_LOWER_BOUND, SKIP_CURSOR_COLOR_HSV_UPPER_BOUND)],
        SKIP_CURSOR_PIXEL_RATIO,
        debug,
        'skip'
    ) != -1
    logger.debug(f'detection.skip: return={result}')
    return result


REPLAY_CURSOR_NUMPY_ROI_TOP_LEFTS = np.array([[760, 705], [760, 1050]])
REPLAY_CURSOR_ROI_WIDTH = 180
REPLAY_CURSOR_ROI_HEIGHT = 70
REPLAY_CURSOR_COLOR_HSV_UPPER_BOUND = (50, 255, 255)
REPLAY_CURSOR_COLOR_HSV_LOWER_BOUND = (30, 100, 150)
REPLAY_CURSOR_PIXEL_RATIO = 0.5


def replay_cursor(img: np.ndarray, debug: Optional[Debugger] = None) -> int:
    pos = util.detect_cursor(
        img,
        REPLAY_CURSOR_NUMPY_ROI_TOP_LEFTS,
        REPLAY_CURSOR_ROI_WIDTH,
        REPLAY_CURSOR_ROI_HEIGHT,
        [(REPLAY_CURSOR_COLOR_HSV_LOWER_BOUND, REPLAY_CURSOR_COLOR_HSV_UPPER_BOUND)],
        REPLAY_CURSOR_PIXEL_RATIO,
        debug,
        'replay_cursor'
    )
    logger.debug(f'detection.replay_cursor: return={pos}')
    return pos


GIVEUP_CURSOR_NUMPY_ROI_TOP_LEFTS = np.array([[760, 705], [760, 1050]])
GIVEUP_CURSOR_ROI_WIDTH = 180
GIVEUP_CURSOR_ROI_HEIGHT = 70
GIVEUP_CURSOR_COLOR_HSV_UPPER_BOUND = (50, 255, 255)
GIVEUP_CURSOR_COLOR_HSV_LOWER_BOUND = (30, 100, 150)
GIVEUP_CURSOR_PIXEL_RATIO = 0.5


def giveup_cursor(img: np.ndarray, debug: Optional[Debugger] = None) -> int:
    pos = util.detect_cursor(
        img,
        GIVEUP_CURSOR_NUMPY_ROI_TOP_LEFTS,
        GIVEUP_CURSOR_ROI_WIDTH,
        GIVEUP_CURSOR_ROI_HEIGHT,
        [(GIVEUP_CURSOR_COLOR_HSV_LOWER_BOUND, GIVEUP_CURSOR_COLOR_HSV_UPPER_BOUND)],
        GIVEUP_CURSOR_PIXEL_RATIO,
        debug,
        'giveup_cursor'
    )
    logger.debug(f'detection.giveup_cursor: return={pos}')
    return pos


LOSE_NUMPY_ROI_TOP_LEFTS = np.array([[660, 320]])
LOSE_ROI_WIDTH = 8
LOSE_ROI_HEIGHT = 8
LOSE_COLOR_HSV_UPPER_BOUND = (50, 255, 255)
LOSE_COLOR_HSV_LOWER_BOUND = (30, 100, 150)
LOSE_PIXEL_RATIO = 0.6

DRAW_NUMPY_ROI_TOP_LEFTS = np.array([[640, 300]])
DRAW_ROI_WIDTH = 8
DRAW_ROI_HEIGHT = 8
DRAW_COLOR_HSV_UPPER_BOUND = (255, 255, 255)
DRAW_COLOR_HSV_LOWER_BOUND = (0, 0, 230)
DRAW_PIXEL_RATIO = 0.6


def result(img: np.ndarray, debug: Optional[Debugger] = None) -> Result:
    ret = Result.Win
    loss = util.detect_cursor(
        img,
        LOSE_NUMPY_ROI_TOP_LEFTS,
        LOSE_ROI_WIDTH,
        LOSE_ROI_HEIGHT,
        [(LOSE_COLOR_HSV_LOWER_BOUND, LOSE_COLOR_HSV_UPPER_BOUND)],
        LOSE_PIXEL_RATIO,
        debug,
        'lose'
    )
    draw = util.detect_cursor(
        img,
        DRAW_NUMPY_ROI_TOP_LEFTS,
        DRAW_ROI_WIDTH,
        DRAW_ROI_HEIGHT,
        [(DRAW_COLOR_HSV_LOWER_BOUND, DRAW_COLOR_HSV_UPPER_BOUND)],
        DRAW_PIXEL_RATIO,
        debug,
        'draw'
    )
    if loss == 0:
        ret = Result.Loss
    if draw == 0:
        ret = Result.Draw
    logger.debug(f'detection.result: return={ret}')
    return ret


LEVEL_CURSOR_NUMPY_ROI_TOP_LEFTS = np.array([[525, 1375]])
LEVEL_CURSOR_ROI_WIDTH = 10
LEVEL_CURSOR_ROI_HEIGHT = 10
LEVEL_CURSOR_COLOR_HSV_UPPER_BOUND = (50, 255, 255)
LEVEL_CURSOR_COLOR_HSV_LOWER_BOUND = (30, 100, 150)
LEVEL_CURSOR_PIXEL_RATIO = 0.5


def level(img: np.ndarray, debug: Optional[Debugger] = None) -> bool:
    pos = util.detect_cursor(
        img,
        LEVEL_CURSOR_NUMPY_ROI_TOP_LEFTS,
        LEVEL_CURSOR_ROI_WIDTH,
        LEVEL_CURSOR_ROI_HEIGHT,
        [(LEVEL_CURSOR_COLOR_HSV_LOWER_BOUND, LEVEL_CURSOR_COLOR_HSV_UPPER_BOUND)],
        LEVEL_CURSOR_PIXEL_RATIO,
        debug,
        'level'
    )
    result = pos == 0
    logger.debug(f'detection.level_cursor: return={result}')
    return result


START_CURSOR_NUMPY_ROI_TOP_LEFTS = np.array([[640, 1430]])
START_CURSOR_ROI_WIDTH = 180
START_CURSOR_ROI_HEIGHT = 70
START_CURSOR_COLOR_HSV_UPPER_BOUND = (50, 255, 255)
START_CURSOR_COLOR_HSV_LOWER_BOUND = (30, 100, 150)
START_CURSOR_PIXEL_RATIO = 0.5


def start(img: np.ndarray, debug: Optional[Debugger] = None) -> bool:
    pos = util.detect_cursor(
        img,
        START_CURSOR_NUMPY_ROI_TOP_LEFTS,
        START_CURSOR_ROI_WIDTH,
        START_CURSOR_ROI_HEIGHT,
        [(START_CURSOR_COLOR_HSV_LOWER_BOUND, START_CURSOR_COLOR_HSV_UPPER_BOUND)],
        START_CURSOR_PIXEL_RATIO,
        debug,
        'start'
    )
    result = pos == 0
    logger.debug(f'detection.start_cursor: return={result}')
    return result
