import cv2
import numpy as np

from logger import logger
from tableturf.manager.detection import util

DECK_CURSOR_ROI_LEFT = 450
DECK_CURSOR_ROI_TOP = 285
DECK_CURSOR_ROI_WIDTH_STEP = 345
DECK_CURSOR_ROI_HEIGHT_STEP = 95
DECK_CURSOR_ROI_WIDTH = 15
DECK_CURSOR_ROI_HEIGHT = 15
DECK_CURSOR_OPENCV_ROI_LEFT_TOPS = np.array([(DECK_CURSOR_ROI_LEFT + DECK_CURSOR_ROI_WIDTH_STEP * (i // 8), DECK_CURSOR_ROI_TOP + DECK_CURSOR_ROI_HEIGHT_STEP * (i % 8)) for i in range(16)])
DECK_CURSOR_NUMPY_ROI_TOP_LEFTS = np.array([util.opencv_to_numpy(idx) for idx in DECK_CURSOR_OPENCV_ROI_LEFT_TOPS])
DECK_CURSOR_COLOR_HSV_UPPER_BOUND = (35, 255, 255)
DECK_CURSOR_COLOR_HSV_LOWER_BOUND = (30, 100, 150)
DECK_CURSOR_PIXEL_RATIO = 0.4


def deck_cursor(img, debug=False) -> int:
    def __cursor_ratios(top_left: np.ndarray) -> float:
        roi = img[top_left[0]:top_left[0] + DECK_CURSOR_ROI_HEIGHT, top_left[1]:top_left[1] + DECK_CURSOR_ROI_WIDTH]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, DECK_CURSOR_COLOR_HSV_LOWER_BOUND, DECK_CURSOR_COLOR_HSV_UPPER_BOUND)
        return np.sum(mask == 255) / (DECK_CURSOR_ROI_WIDTH * DECK_CURSOR_ROI_HEIGHT)

    ratios = np.array([__cursor_ratios(top_left) for top_left in DECK_CURSOR_NUMPY_ROI_TOP_LEFTS])
    pos = np.argmax(ratios)
    if ratios[pos] < DECK_CURSOR_PIXEL_RATIO:
        pos = -1
    if debug:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, DECK_CURSOR_COLOR_HSV_LOWER_BOUND, DECK_CURSOR_COLOR_HSV_UPPER_BOUND)
        mask = cv2.merge((mask, mask, mask))
        for i, roi in enumerate(DECK_CURSOR_OPENCV_ROI_LEFT_TOPS):
            cv2.rectangle(img, roi, roi + (DECK_CURSOR_ROI_WIDTH, DECK_CURSOR_ROI_HEIGHT), (0, 255, 0), 1)
            cv2.rectangle(mask, roi, roi + (DECK_CURSOR_ROI_WIDTH, DECK_CURSOR_ROI_HEIGHT), (0, 255, 0), 1)
            cv2.putText(mask, f'{ratios[i]:.3}', roi + (0,-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(mask, f'{i}', roi + np.array((DECK_CURSOR_ROI_WIDTH / 5, DECK_CURSOR_ROI_HEIGHT / 1.5), dtype=int), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        util.show(img)
        util.show(mask)
    logger.debug(f'detection.deck_cursor: return={pos}')
    return pos
