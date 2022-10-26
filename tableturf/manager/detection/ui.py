import numpy as np

from logger import logger
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
DECK_CURSOR_PIXEL_RATIO = 0.4


def deck_cursor(img, debug=False) -> int:
    pos = util.detect_cursor(
        img,
        DECK_CURSOR_NUMPY_ROI_TOP_LEFTS,
        DECK_CURSOR_ROI_WIDTH,
        DECK_CURSOR_ROI_HEIGHT,
        DECK_CURSOR_COLOR_HSV_LOWER_BOUND,
        DECK_CURSOR_COLOR_HSV_UPPER_BOUND,
        DECK_CURSOR_PIXEL_RATIO,
        debug,
    )
    logger.debug(f'detection.deck_cursor: return={pos}')
    return pos


HANDS_CURSOR_NUMPY_ROI_TOP_LEFTS = np.array([[470, 14], [480, 266], [793, 14], [797, 266], [909, 14], [898, 266]])
HANDS_CURSOR_ROI_WIDTH = 30
HANDS_CURSOR_ROI_HEIGHT = 30
HANDS_CURSOR_COLOR_HSV_UPPER_BOUND = (35, 255, 255)
HANDS_CURSOR_COLOR_HSV_LOWER_BOUND = (30, 100, 150)
HANDS_CURSOR_PIXEL_RATIO = 0.3


def hands_cursor(img: np.ndarray, debug=False) -> int:
    pos = util.detect_cursor(
        img,
        HANDS_CURSOR_NUMPY_ROI_TOP_LEFTS,
        HANDS_CURSOR_ROI_WIDTH,
        HANDS_CURSOR_ROI_HEIGHT,
        HANDS_CURSOR_COLOR_HSV_LOWER_BOUND,
        HANDS_CURSOR_COLOR_HSV_UPPER_BOUND,
        HANDS_CURSOR_PIXEL_RATIO,
        debug,
    )
    logger.debug(f'detection.hands_cursor: return={pos}')
    return pos


REDRAW_CURSOR_NUMPY_ROI_TOP_LEFTS = np.array([[760, 900], [760, 1250]])
REDRAW_CURSOR_ROI_WIDTH = 180
REDRAW_CURSOR_ROI_HEIGHT = 70
REDRAW_CURSOR_COLOR_HSV_UPPER_BOUND = (50, 255, 255)
REDRAW_CURSOR_COLOR_HSV_LOWER_BOUND = (30, 100, 150)
REDRAW_CURSOR_PIXEL_RATIO = 0.5


def redraw_cursor(img: np.ndarray, debug=False) -> int:
    pos = util.detect_cursor(
        img,
        REDRAW_CURSOR_NUMPY_ROI_TOP_LEFTS,
        REDRAW_CURSOR_ROI_WIDTH,
        REDRAW_CURSOR_ROI_HEIGHT,
        REDRAW_CURSOR_COLOR_HSV_LOWER_BOUND,
        REDRAW_CURSOR_COLOR_HSV_UPPER_BOUND,
        REDRAW_CURSOR_PIXEL_RATIO,
        debug,
    )
    logger.debug(f'detection.redraw_cursor: return={pos}')
    return pos


SPECIAL_ON_CURSOR_NUMPY_ROI_TOP_LEFTS = np.array([[840, 295]])
SPECIAL_ON_CURSOR_ROI_WIDTH = 200
SPECIAL_ON_CURSOR_ROI_HEIGHT = 50
SPECIAL_ON_CURSOR_COLOR_HSV_UPPER_BOUND = (160, 160, 160)
SPECIAL_ON_CURSOR_COLOR_HSV_LOWER_BOUND = (0, 0, 0)
SPECIAL_ON_CURSOR_PIXEL_RATIO = 0.8


def special_on(img: np.ndarray, debug=False) -> bool:
    result = util.detect_cursor(
        img,
        SPECIAL_ON_CURSOR_NUMPY_ROI_TOP_LEFTS,
        SPECIAL_ON_CURSOR_ROI_WIDTH,
        SPECIAL_ON_CURSOR_ROI_HEIGHT,
        SPECIAL_ON_CURSOR_COLOR_HSV_LOWER_BOUND,
        SPECIAL_ON_CURSOR_COLOR_HSV_UPPER_BOUND,
        SPECIAL_ON_CURSOR_PIXEL_RATIO,
        debug,
    ) != -1
    logger.debug(f'detection.special_on: return={result}')
    return result
