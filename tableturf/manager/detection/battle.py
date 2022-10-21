from typing import List

import cv2
import numpy as np

from logger import logger
from tableturf.manager.detection import util
from tableturf.model import Card, Grid

# card grid
HANDS_GRID_ROI_LEFTS = [51, 51, 305, 305]
HANDS_GRID_ROI_TOPS = [192, 516, 203, 516]
HANDS_GRID_ROI_WIDTH = 14
HANDS_GRID_ROI_HEIGHT = 14
HANDS_GRID_ROI_WIDTH_STEPS = [25.9, 25.9, 24.7, 24.7]
HANDS_GRID_ROI_HEIGHT_STEPS = [23.9, 23.9, 23, 23]
HANDS_GRID_ROI_HEIGHT_OFFSETS = [1, 0, 0.8, 0]
HANDS_GRID_OPENCV_ROI_LEFT_TOP = np.stack([HANDS_GRID_ROI_LEFTS, HANDS_GRID_ROI_TOPS], axis=1)  # shape: (4, 2)
HANDS_GRID_NUMPY_ROI_TOP_LEFT = np.array([util.opencv_to_numpy(idx) for idx in HANDS_GRID_OPENCV_ROI_LEFT_TOP])  # shape: (4, 2)
HANDS_GRID_OPENCV_ROI_LEFT_TOPS = np.array([(top_left[0] + int(HANDS_GRID_ROI_WIDTH_STEPS[i] * (k % 8)), top_left[1] + int(HANDS_GRID_ROI_HEIGHT_STEPS[i] * (k // 8)) + int(HANDS_GRID_ROI_HEIGHT_OFFSETS[i] * (k % 8))) for i, top_left in enumerate(HANDS_GRID_OPENCV_ROI_LEFT_TOP) for k in range(64)]).reshape((4, 64, 2))  # shape: (4, 64, 2)
HANDS_GRID_NUMPY_ROI_TOP_LEFTS = np.array([util.opencv_to_numpy(idx) for grid in HANDS_GRID_OPENCV_ROI_LEFT_TOPS for idx in grid]).reshape((4, 64, 2))  # shape: (4, 64, 2)
# card cost
HANDS_COST_ROI_LEFTS = [123, 123, 373, 373]
HANDS_COST_ROI_TOPS = [410, 731, 414, 725]
HANDS_COST_ROI_WIDTH = 12
HANDS_COST_ROI_HEIGHT = 12
HANDS_COST_ROI_WIDTH_STEPS = [22.9, 22.9, 21.5, 21.5]
HANDS_COST_ROI_HEIGHT_OFFSETS = [-0.5, -0.6, -0.5, -0.6]
HANDS_COST_OPENCV_ROI_LEFT_TOP = np.stack([HANDS_COST_ROI_LEFTS, HANDS_COST_ROI_TOPS], axis=1)  # shape: (4, 2)
HANDS_COST_NUMPY_ROI_TOP_LEFT = np.array([util.opencv_to_numpy(idx) for idx in HANDS_COST_OPENCV_ROI_LEFT_TOP])  # shape: (4, 2)
HANDS_COST_OPENCV_ROI_LEFT_TOPS = np.array([(top_left[0] + int(HANDS_COST_ROI_WIDTH_STEPS[i] * k), top_left[1] + int(HANDS_COST_ROI_HEIGHT_OFFSETS[i] * k)) for i, top_left in enumerate(HANDS_COST_OPENCV_ROI_LEFT_TOP) for k in range(6)]).reshape((4, 6, 2))  # shape: (4, 6, 2)
HANDS_COST_NUMPY_ROI_TOP_LEFTS = np.array([util.opencv_to_numpy(idx) for grid in HANDS_COST_OPENCV_ROI_LEFT_TOPS for idx in grid]).reshape((4, 6, 2))  # shape: (4, 6, 2)

MY_INK_COLOR_HSV_UPPER_BOUND = (35, 255, 255)
MY_INK_COLOR_HSV_LOWER_BOUND = (30, 150, 150)
MY_SPECIAL_COLOR_HSV_UPPER_BOUND = (25, 255, 255)
MY_SPECIAL_COLOR_HSV_LOWER_BOUND = (20, 150, 150)
GRID_PIXEL_RATIO = 0.8


def hands(img, debug=False) -> int:
    def __grid_ratios(top_left: np.ndarray, lower_bound, upper_bound) -> List[Card]:
        roi = img[top_left[0]:top_left[0] + HANDS_GRID_ROI_HEIGHT, top_left[1]:top_left[1] + HANDS_GRID_ROI_WIDTH]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        return np.sum(mask == 255) / (HANDS_GRID_ROI_WIDTH * HANDS_GRID_ROI_HEIGHT)

    grid_ink_ratios = np.array([__grid_ratios(idx, MY_INK_COLOR_HSV_LOWER_BOUND, MY_INK_COLOR_HSV_UPPER_BOUND) for grid in HANDS_GRID_NUMPY_ROI_TOP_LEFTS for idx in grid]).reshape(4, 64)
    grid_special_ratios = np.array([__grid_ratios(idx, MY_SPECIAL_COLOR_HSV_LOWER_BOUND, MY_SPECIAL_COLOR_HSV_UPPER_BOUND) for grid in HANDS_GRID_NUMPY_ROI_TOP_LEFTS for idx in grid]).reshape(4, 64)
    cost_ratios = np.array([__grid_ratios(idx, MY_SPECIAL_COLOR_HSV_LOWER_BOUND, MY_SPECIAL_COLOR_HSV_UPPER_BOUND) for grid in HANDS_COST_NUMPY_ROI_TOP_LEFTS for idx in grid]).reshape(4, 6)
    grids = np.zeros((4, 64), dtype=int)
    grids[grid_ink_ratios > GRID_PIXEL_RATIO] = Grid.MyInk.value
    grids[grid_special_ratios > GRID_PIXEL_RATIO] = Grid.MySpecial.value
    grids = grids.reshape((4,8,8))
    costs = np.sum(cost_ratios > GRID_PIXEL_RATIO, axis=1)

    if debug:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ink_mask = cv2.inRange(hsv, MY_INK_COLOR_HSV_LOWER_BOUND, MY_INK_COLOR_HSV_UPPER_BOUND)
        special_mask = cv2.inRange(hsv, MY_SPECIAL_COLOR_HSV_LOWER_BOUND, MY_SPECIAL_COLOR_HSV_UPPER_BOUND)
        mask = np.maximum(ink_mask, special_mask)
        mask = cv2.merge((mask, mask, mask))
        for i, grid in enumerate(HANDS_GRID_OPENCV_ROI_LEFT_TOPS):
            for k, roi in enumerate(grid):
                cv2.rectangle(img, roi, roi + (HANDS_GRID_ROI_WIDTH, HANDS_GRID_ROI_HEIGHT), (0, 255, 0), 1)
                cv2.rectangle(mask, roi, roi + (HANDS_GRID_ROI_WIDTH, HANDS_GRID_ROI_HEIGHT), (0, 255, 0), 1)
                cv2.putText(mask, f'{k}', roi + np.array((HANDS_GRID_ROI_WIDTH / 5, HANDS_GRID_ROI_HEIGHT / 1.5), dtype=int), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                if grid_ink_ratios[i][k] > GRID_PIXEL_RATIO:
                    cv2.rectangle(mask, roi, roi + (HANDS_GRID_ROI_WIDTH, HANDS_GRID_ROI_HEIGHT), (0, 0, 255), 1)
                    cv2.putText(mask, f'{k}', roi + np.array((HANDS_GRID_ROI_WIDTH / 5, HANDS_GRID_ROI_HEIGHT / 1.5), dtype=int), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                if grid_special_ratios[i][k] > GRID_PIXEL_RATIO:
                    cv2.rectangle(mask, roi, roi + (HANDS_GRID_ROI_WIDTH, HANDS_GRID_ROI_HEIGHT), (255, 0, 0), 1)
                    cv2.putText(mask, f'{k}', roi + np.array((HANDS_GRID_ROI_WIDTH / 5, HANDS_GRID_ROI_HEIGHT / 1.5), dtype=int), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        for i, cost in enumerate(HANDS_COST_OPENCV_ROI_LEFT_TOPS):
            for k, roi in enumerate(cost):
                cv2.rectangle(img, roi, roi + (HANDS_COST_ROI_WIDTH, HANDS_COST_ROI_HEIGHT), (0, 255, 0), 1)
                cv2.rectangle(mask, roi, roi + (HANDS_COST_ROI_WIDTH, HANDS_COST_ROI_HEIGHT), (0, 255, 0), 1)
            cv2.putText(mask, f'{costs[i]}', HANDS_COST_OPENCV_ROI_LEFT_TOP[i] + np.array((-HANDS_GRID_ROI_WIDTH * 1.5, HANDS_COST_ROI_HEIGHT), dtype=int), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        util.show(img)
        util.show(mask)
    cards = [Card(grids[i], costs[i]) for i in range(4)]
    logger.debug(f'detection.hands: return={cards}')
    return cards
