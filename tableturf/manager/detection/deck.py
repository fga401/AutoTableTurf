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
