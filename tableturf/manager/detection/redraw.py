import cv2
import numpy as np

from logger import logger
from tableturf.manager.detection import util

REDRAW_CURSOR_COLOR_HSV_UPPER_BOUND = (50, 255, 255)
REDRAW_CURSOR_COLOR_HSV_LOWER_BOUND = (30, 100, 150)
REDRAW_CURSOR_OPENCV_ROI_LEFT_TOPS = np.array([[900, 760], [1250, 760]])
REDRAW_CURSOR_NUMPY_ROI_TOP_LEFTS = np.array([util.opencv_to_numpy(idx) for idx in REDRAW_CURSOR_OPENCV_ROI_LEFT_TOPS])
REDRAW_CURSOR_ROI_WIDTH = 180
REDRAW_CURSOR_ROI_HEIGHT = 70
REDRAW_CURSOR_PIXEL_RATIO = 0.5


def redraw_cursor(img: np.ndarray, debug=False) -> int:
    def __cursor_ratios(top_left: np.ndarray) -> float:
        roi = img[top_left[0]:top_left[0] + REDRAW_CURSOR_ROI_HEIGHT, top_left[1]:top_left[1] + REDRAW_CURSOR_ROI_WIDTH]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, REDRAW_CURSOR_COLOR_HSV_LOWER_BOUND, REDRAW_CURSOR_COLOR_HSV_UPPER_BOUND)
        return np.sum(mask == 255) / (REDRAW_CURSOR_ROI_WIDTH * REDRAW_CURSOR_ROI_HEIGHT)

    ratios = np.array([__cursor_ratios(top_left) for top_left in REDRAW_CURSOR_NUMPY_ROI_TOP_LEFTS])
    pos = np.argmax(ratios)
    if ratios[pos] < REDRAW_CURSOR_PIXEL_RATIO:
        pos = -1
    if debug:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, REDRAW_CURSOR_COLOR_HSV_LOWER_BOUND, REDRAW_CURSOR_COLOR_HSV_UPPER_BOUND)
        mask = cv2.merge((mask, mask, mask))
        for i, roi in enumerate(REDRAW_CURSOR_OPENCV_ROI_LEFT_TOPS):
            cv2.rectangle(img, roi, roi + (REDRAW_CURSOR_ROI_WIDTH, REDRAW_CURSOR_ROI_HEIGHT), (0, 255, 0), 3)
            cv2.rectangle(mask, roi, roi + (REDRAW_CURSOR_ROI_WIDTH, REDRAW_CURSOR_ROI_HEIGHT), (0, 255, 0), 3)
            cv2.putText(mask, f'{ratios[i]:.3}', roi + (0, -10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(mask, f'{i}', roi + np.array((REDRAW_CURSOR_ROI_WIDTH / 5, REDRAW_CURSOR_ROI_HEIGHT / 1.5), dtype=int), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        util.show(img)
        util.show(mask)
    logger.debug(f'detection.redraw_cursor: return={pos}')
    return pos
