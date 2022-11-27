from typing import Optional

import numpy as np

from logger import logger
from tableturf.debugger.interface import Debugger
from tableturf.manager.detection import util

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
