import cv2
import numpy as np

from logger import logger
from tableturf.manager.detection import util

CURSOR_ROI_TOP_LEFT_X = 450
CURSOR_ROI_TOP_LEFT_Y = 285
CURSOR_ROI_WIDTH_STEP = 345
CURSOR_ROI_HEIGHT_STEP = 95
CURSOR_ROI_WIDTH = 15
CURSOR_ROI_HEIGHT = 15
CURSOR_OPENCV_ROIS = np.array([np.array((CURSOR_ROI_TOP_LEFT_X, CURSOR_ROI_TOP_LEFT_Y)) + (CURSOR_ROI_WIDTH_STEP * (i // 8), CURSOR_ROI_HEIGHT_STEP * (i % 8)) for i in range(16)])
CURSOR_NUMPY_ROIS = np.array([np.array((CURSOR_ROI_TOP_LEFT_Y, CURSOR_ROI_TOP_LEFT_X)) + (CURSOR_ROI_HEIGHT_STEP * (i % 8), CURSOR_ROI_WIDTH_STEP * (i // 8)) for i in range(16)])
CURSOR_COLOR_HSV_UPPER_BOUND = (35, 255, 255)
CURSOR_COLOR_HSV_LOWER_BOUND = (30, 100, 150)
CURSOR_PIXEL_RATIO = 0.4


def deck_cursor(img, debug=False) -> int:
    def __cursor_ratios(top_left: np.ndarray) -> float:
        roi = img[top_left[0]:top_left[0] + CURSOR_ROI_HEIGHT, top_left[1]:top_left[1] + CURSOR_ROI_WIDTH]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, CURSOR_COLOR_HSV_LOWER_BOUND, CURSOR_COLOR_HSV_UPPER_BOUND)
        return np.sum(mask == 255) / (CURSOR_ROI_WIDTH * CURSOR_ROI_HEIGHT)

    ratios = np.apply_along_axis(__cursor_ratios, axis=1, arr=CURSOR_NUMPY_ROIS)
    pos = np.argmax(ratios)
    if ratios[pos] < CURSOR_PIXEL_RATIO:
        pos = -1
    if debug:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, CURSOR_COLOR_HSV_LOWER_BOUND, CURSOR_COLOR_HSV_UPPER_BOUND)
        mask = cv2.merge((mask, mask, mask))
        for i, roi in enumerate(CURSOR_OPENCV_ROIS):
            cv2.rectangle(img, roi, roi + (CURSOR_ROI_WIDTH, CURSOR_ROI_HEIGHT), (0, 255, 0), 1)
            cv2.rectangle(mask, roi, roi + (CURSOR_ROI_WIDTH, CURSOR_ROI_HEIGHT), (0, 255, 0), 1)
            cv2.putText(mask, f'{ratios[i]:.3}', roi, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        util.show(img)
        util.show(mask)
    logger.debug(f'detection.deck_cursor: return={pos}')
    return pos
