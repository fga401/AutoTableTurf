from typing import List

import cv2
import numpy as np

from logger import logger
from tableturf.manager.detection import util
from tableturf.manager.detection.ui import hands_cursor
from tableturf.model import Card, Grid

# card grid
HANDS_GRID_TOP_LEFTS = np.array([[192, 51], [516, 51], [203, 305], [516, 305]])
HANDS_GRID_ROI_WIDTH = 14
HANDS_GRID_ROI_HEIGHT = 14
HANDS_GRID_ROI_WIDTH_STEPS = [25.9, 25.9, 24.7, 24.7]
HANDS_GRID_ROI_HEIGHT_STEPS = [23.9, 23.9, 23, 23]
HANDS_GRID_ROI_HEIGHT_OFFSETS = [1, 0, 0.8, 0]
HANDS_GRID_NUMPY_ROI_TOP_LEFTS = np.array([util.grid_roi_top_lefts(top_left, 8, 8, HANDS_GRID_ROI_WIDTH_STEPS[i], HANDS_GRID_ROI_HEIGHT_STEPS[i], 0, HANDS_GRID_ROI_HEIGHT_OFFSETS[i]) for i, top_left in enumerate(HANDS_GRID_TOP_LEFTS)]).reshape((4, 64, 2))
# card cost
HANDS_COST_TOP_LEFTS = np.array([[410, 123], [731, 123], [414, 373], [725, 373]])
HANDS_COST_ROI_WIDTH = 12
HANDS_COST_ROI_HEIGHT = 12
HANDS_COST_ROI_WIDTH_STEPS = [22.9, 22.9, 21.5, 21.5]
HANDS_COST_ROI_HEIGHT_OFFSETS = [0, -0.6, 0, -0.6]
HANDS_COST_OPENCV_ROI_LEFT_TOP = np.array([util.numpy_to_opencv(idx) for idx in HANDS_COST_TOP_LEFTS])  # shape: (4, 2)
HANDS_COST_NUMPY_ROI_TOP_LEFTS = np.array([util.grid_roi_top_lefts(top_left, 6, 1, HANDS_COST_ROI_WIDTH_STEPS[i], 0, 0, HANDS_COST_ROI_HEIGHT_OFFSETS[i]) for i, top_left in enumerate(HANDS_COST_TOP_LEFTS)]).reshape((4, 6, 2))

# focus card grid
FOCUS_GRID_TOP_LEFTS = np.array([[192, 42], [514, 43], [200, 295], [515, 295]])
FOCUS_GRID_ROI_WIDTH = 14
FOCUS_GRID_ROI_HEIGHT = 14
FOCUS_GRID_ROI_WIDTH_STEPS = [26.9, 26.9, 25.7, 25.7]
FOCUS_GRID_ROI_HEIGHT_STEPS = [24.9, 24.9, 24, 24]
FOCUS_GRID_ROI_WIDTH_OFFSETS = [1, 1, 1, 1]
FOCUS_GRID_ROI_HEIGHT_OFFSETS = [-0.6, -1, -0.4, -1]
FOCUS_GRID_NUMPY_ROI_TOP_LEFTS = np.array([util.grid_roi_top_lefts(top_left, 8, 8, FOCUS_GRID_ROI_WIDTH_STEPS[i], FOCUS_GRID_ROI_HEIGHT_STEPS[i], FOCUS_GRID_ROI_WIDTH_OFFSETS[i], FOCUS_GRID_ROI_HEIGHT_OFFSETS[i]) for i, top_left in enumerate(FOCUS_GRID_TOP_LEFTS)]).reshape((4, 64, 2))
# focus card cost
FOCUS_COST_TOP_LEFTS = np.array([[415, 128], [735, 127], [417, 375], [730, 376]])
FOCUS_COST_ROI_WIDTH = 12
FOCUS_COST_ROI_HEIGHT = 12
FOCUS_COST_ROI_WIDTH_STEPS = [22.9, 23.9, 22.5, 22.5]
FOCUS_COST_ROI_HEIGHT_OFFSETS = [-0.5, -1, 0, -1]
FOCUS_COST_NUMPY_ROI_TOP_LEFTS = np.array([util.grid_roi_top_lefts(top_left, 6, 1, FOCUS_COST_ROI_WIDTH_STEPS[i], 0, 0, FOCUS_COST_ROI_HEIGHT_OFFSETS[i]) for i, top_left in enumerate(FOCUS_COST_TOP_LEFTS)]).reshape((4, 6, 2))

MY_INK_COLOR_HSV_UPPER_BOUND = (35, 255, 255)
MY_INK_COLOR_HSV_LOWER_BOUND = (30, 150, 150)
MY_SPECIAL_COLOR_HSV_UPPER_BOUND = (25, 255, 255)
MY_SPECIAL_COLOR_HSV_LOWER_BOUND = (20, 150, 150)
MY_INK_GRAY_COLOR_HSV_UPPER_BOUND = (35, 255, 255)
MY_INK_GRAY_COLOR_HSV_LOWER_BOUND = (25, 43, 43)
MY_SPECIAL_GRAY_COLOR_HSV_UPPER_BOUND = (25, 255, 255)
MY_SPECIAL_GRAY_COLOR_HSV_LOWER_BOUND = (0, 0, 0)
GRID_PIXEL_RATIO = 0.6


def hands(img, cursor=None, debug=False) -> List[Card]:
    def __grid_ratios(top_left: np.ndarray, lower_bound, upper_bound) -> List[Card]:
        roi = img[top_left[0]:top_left[0] + HANDS_GRID_ROI_HEIGHT, top_left[1]:top_left[1] + HANDS_GRID_ROI_WIDTH]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        return np.sum(mask == 255) / (HANDS_GRID_ROI_WIDTH * HANDS_GRID_ROI_HEIGHT)

    if cursor is None:
        cursor = hands_cursor(img, debug)
    grid_rois = HANDS_GRID_NUMPY_ROI_TOP_LEFTS.copy()
    cost_rois = HANDS_COST_NUMPY_ROI_TOP_LEFTS.copy()
    if 0 <= cursor < 4:
        grid_rois[cursor] = FOCUS_GRID_NUMPY_ROI_TOP_LEFTS[cursor]
        cost_rois[cursor] = FOCUS_COST_NUMPY_ROI_TOP_LEFTS[cursor]

    ink_lower_bound, ink_upper_bound = MY_INK_COLOR_HSV_LOWER_BOUND, MY_INK_COLOR_HSV_UPPER_BOUND
    special_lower_bound, special_upper_bound = MY_SPECIAL_COLOR_HSV_LOWER_BOUND, MY_SPECIAL_COLOR_HSV_UPPER_BOUND
    grid_ink_ratios = np.array([__grid_ratios(idx, ink_lower_bound, ink_upper_bound) for grid in grid_rois for idx in grid]).reshape(4, 64)
    grid_special_ratios = np.array([__grid_ratios(idx, special_lower_bound, special_upper_bound) for grid in grid_rois for idx in grid]).reshape(4, 64)
    if grid_ink_ratios.max() < GRID_PIXEL_RATIO and grid_special_ratios.max() < GRID_PIXEL_RATIO:
        ink_lower_bound, ink_upper_bound = MY_INK_GRAY_COLOR_HSV_LOWER_BOUND, MY_INK_GRAY_COLOR_HSV_UPPER_BOUND
        special_lower_bound, special_upper_bound = MY_SPECIAL_GRAY_COLOR_HSV_LOWER_BOUND, MY_SPECIAL_GRAY_COLOR_HSV_UPPER_BOUND
        grid_ink_ratios = np.array([__grid_ratios(idx, ink_lower_bound, ink_upper_bound) for grid in grid_rois for idx in grid]).reshape(4, 64)
        grid_special_ratios = np.array([__grid_ratios(idx, special_lower_bound, special_upper_bound) for grid in grid_rois for idx in grid]).reshape(4, 64)
    cost_ratios = np.array([__grid_ratios(idx, special_lower_bound, special_upper_bound) for grid in cost_rois for idx in grid]).reshape(4, 6)

    grids = np.zeros((4, 64), dtype=int)
    grids[grid_ink_ratios > GRID_PIXEL_RATIO] = Grid.MyInk.value
    grids[grid_special_ratios > GRID_PIXEL_RATIO] = Grid.MySpecial.value
    grids = grids.reshape((4, 8, 8))
    costs = np.sum(cost_ratios > GRID_PIXEL_RATIO, axis=1)

    if debug:
        grid_rois = np.array([util.numpy_to_opencv(idx) for grid in grid_rois for idx in grid]).reshape((4, 64, 2))
        cost_rois = np.array([util.numpy_to_opencv(idx) for grid in cost_rois for idx in grid]).reshape((4, 6, 2))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ink_mask = cv2.inRange(hsv, ink_lower_bound, ink_upper_bound)
        special_mask = cv2.inRange(hsv, special_lower_bound, special_upper_bound)
        mask = np.maximum(ink_mask, special_mask)
        mask = cv2.merge((mask, mask, mask))
        for i, grid in enumerate(grid_rois):
            for k, roi in enumerate(grid):
                cv2.rectangle(img, roi, roi + (HANDS_GRID_ROI_WIDTH, HANDS_GRID_ROI_HEIGHT), (0, 255, 0), 1)
                cv2.rectangle(mask, roi, roi + (HANDS_GRID_ROI_WIDTH, HANDS_GRID_ROI_HEIGHT), (0, 255, 0), 1)
                cv2.putText(mask, f'{k}', roi + np.rint([HANDS_GRID_ROI_WIDTH / 10, HANDS_GRID_ROI_HEIGHT / 1.3]).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                if grid_ink_ratios[i][k] > GRID_PIXEL_RATIO:
                    cv2.rectangle(mask, roi, roi + (HANDS_GRID_ROI_WIDTH, HANDS_GRID_ROI_HEIGHT), (0, 0, 255), 1)
                    cv2.putText(mask, f'{k}', roi + np.rint([HANDS_GRID_ROI_WIDTH / 10, HANDS_GRID_ROI_HEIGHT / 1.3]).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                if grid_special_ratios[i][k] > GRID_PIXEL_RATIO:
                    cv2.rectangle(mask, roi, roi + (HANDS_GRID_ROI_WIDTH, HANDS_GRID_ROI_HEIGHT), (255, 0, 0), 1)
                    cv2.putText(mask, f'{k}', roi + np.rint([HANDS_GRID_ROI_WIDTH / 10, HANDS_GRID_ROI_HEIGHT / 1.3]).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        for i, cost in enumerate(cost_rois):
            for k, roi in enumerate(cost):
                cv2.rectangle(img, roi, roi + (HANDS_COST_ROI_WIDTH, HANDS_COST_ROI_HEIGHT), (0, 255, 0), 1)
                cv2.rectangle(mask, roi, roi + (HANDS_COST_ROI_WIDTH, HANDS_COST_ROI_HEIGHT), (0, 255, 0), 1)
            cv2.putText(mask, f'{costs[i]}', HANDS_COST_OPENCV_ROI_LEFT_TOP[i] + np.rint([-HANDS_GRID_ROI_WIDTH * 1.5, HANDS_COST_ROI_HEIGHT]).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        util.show(img)
        util.show(mask)
    cards = [Card(grids[i], costs[i]) for i in range(4)]
    logger.debug(f'detection.hands: return={cards}')
    return cards
