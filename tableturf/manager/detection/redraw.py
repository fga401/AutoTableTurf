import cv2
import numpy as np

from logger import logger
from tableturf.manager.detection import util

REDRAW_CURSOR_OPENCV_ROI_LEFT_TOPS = np.array([[900, 760], [1250, 760]])
REDRAW_CURSOR_NUMPY_ROI_TOP_LEFTS = np.array([util.opencv_to_numpy(idx) for idx in REDRAW_CURSOR_OPENCV_ROI_LEFT_TOPS])
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
