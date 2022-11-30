from typing import List, Optional

import cv2
import numpy as np

from logger import logger
from tableturf.debugger.interface import Debugger
from tableturf.manager.detection import util
from tableturf.model import Card, Grid

# deck grid
DECK_GRID_TOP_LEFTS = np.array([[102, 1357], [99, 1493], [96, 1629], [282, 1357], [280, 1492], [278, 1628], [463, 1357], [462, 1491], [461, 1628], [643, 1357], [644, 1492], [644, 1629], [824, 1357], [826, 1492], [827, 1629]])
DECK_GRID_ROI_WIDTH = 8
DECK_GRID_ROI_HEIGHT = 8
DECK_GRID_ROI_WIDTH_STEPS = [14, 14.3, 14.6]
DECK_GRID_ROI_HEIGHT_OFFSETS = [-0.5, -0.3, 0, 0, 0.2]
DECK_GRID_NUMPY_ROI_TOP_LEFTS = np.array([util.grid_roi_top_lefts(top_left, 8, 8, DECK_GRID_ROI_WIDTH_STEPS[i % 3], 14, 0, DECK_GRID_ROI_HEIGHT_OFFSETS[i // 3]) for i, top_left in enumerate(DECK_GRID_TOP_LEFTS)]).reshape((15, 64, 2))
# deck cost
DECK_COST_TOP_LEFTS = np.array([[227, 1395], [225, 1531], [223, 1668], [408, 1395], [407, 1531], [406, 1668], [589, 1395], [589, 1531], [589, 1668], [770, 1395], [771, 1531], [773, 1668], [950, 1395], [953, 1531], [956, 1668]])
DECK_COST_ROI_WIDTH = 8
DECK_COST_ROI_HEIGHT = 8
DECK_COST_ROI_HEIGHT_OFFSETS = [-0.3, 0, 0, 0, 0.3]
DECK_COST_OPENCV_ROI_LEFT_TOP = np.array([util.numpy_to_opencv(idx) for idx in DECK_COST_TOP_LEFTS])  # shape: (15, 2)
DECK_COST_NUMPY_ROI_TOP_LEFTS = np.array([util.grid_roi_top_lefts(top_left, 6, 1, 13, 0, 0, DECK_COST_ROI_HEIGHT_OFFSETS[i // 3]) for i, top_left in enumerate(DECK_COST_TOP_LEFTS)]).reshape((15, 6, 2))

MY_INK_LIGHTER_COLOR_HSV_UPPER_BOUND = (35, 255, 255)
MY_INK_LIGHTER_COLOR_HSV_LOWER_BOUND = (30, 100, 150)
MY_SPECIAL_LIGHTER_COLOR_HSV_UPPER_BOUND = (25, 255, 255)
MY_SPECIAL_LIGHTER_COLOR_HSV_LOWER_BOUND = (20, 100, 150)
MY_INK_DARKER_COLOR_HSV_UPPER_BOUND = (35, 255, 140)
MY_INK_DARKER_COLOR_HSV_LOWER_BOUND = (27, 100, 100)
MY_SPECIAL_DARKER_COLOR_HSV_UPPER_BOUND = (27, 255, 140)
MY_SPECIAL_DARKER_COLOR_HSV_LOWER_BOUND = (15, 100, 100)
GRID_PIXEL_RATIO = 0.3


def deck(img: np.ndarray, debug: Optional[Debugger] = None) -> List[Card]:
    def __grid_ratios(top_left: np.ndarray, lower_bound, upper_bound) -> List[Card]:
        roi = img[top_left[0]:top_left[0] + DECK_GRID_ROI_HEIGHT, top_left[1]:top_left[1] + DECK_GRID_ROI_WIDTH]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        return np.sum(mask == 255) / (DECK_GRID_ROI_WIDTH * DECK_GRID_ROI_HEIGHT)

    ink_lower_bound, ink_upper_bound = MY_INK_LIGHTER_COLOR_HSV_LOWER_BOUND, MY_INK_LIGHTER_COLOR_HSV_UPPER_BOUND
    special_lower_bound, special_upper_bound = MY_SPECIAL_LIGHTER_COLOR_HSV_LOWER_BOUND, MY_SPECIAL_LIGHTER_COLOR_HSV_UPPER_BOUND
    grid_ink_ratios = np.array([__grid_ratios(idx, ink_lower_bound, ink_upper_bound) for grid in DECK_GRID_NUMPY_ROI_TOP_LEFTS for idx in grid]).reshape(15, 64)
    grid_special_ratios = np.array([__grid_ratios(idx, special_lower_bound, special_upper_bound) for grid in DECK_GRID_NUMPY_ROI_TOP_LEFTS for idx in grid]).reshape(15, 64)
    dark_grid_ink_ratios = np.array([__grid_ratios(idx, MY_INK_DARKER_COLOR_HSV_LOWER_BOUND, MY_INK_DARKER_COLOR_HSV_UPPER_BOUND) for grid in DECK_GRID_NUMPY_ROI_TOP_LEFTS for idx in grid]).reshape(15, 64)
    dark_grid_special_ratios = np.array([__grid_ratios(idx, MY_SPECIAL_DARKER_COLOR_HSV_LOWER_BOUND, MY_SPECIAL_DARKER_COLOR_HSV_UPPER_BOUND) for grid in DECK_GRID_NUMPY_ROI_TOP_LEFTS for idx in grid]).reshape(15, 64)

    cost_ratios = np.array([__grid_ratios(idx, special_lower_bound, special_upper_bound) for grid in DECK_COST_NUMPY_ROI_TOP_LEFTS for idx in grid]).reshape(15, 6)
    dark_cost_ratios = np.array([__grid_ratios(idx, MY_SPECIAL_DARKER_COLOR_HSV_LOWER_BOUND, MY_SPECIAL_DARKER_COLOR_HSV_UPPER_BOUND) for grid in DECK_COST_NUMPY_ROI_TOP_LEFTS for idx in grid]).reshape(15, 6)
    cost_ratios = np.maximum(cost_ratios, dark_cost_ratios)

    grids = np.zeros((15, 64), dtype=int)
    grids[grid_ink_ratios > GRID_PIXEL_RATIO] = Grid.MyInk.value
    grids[grid_special_ratios > GRID_PIXEL_RATIO] = Grid.MySpecial.value
    grids[dark_grid_ink_ratios > GRID_PIXEL_RATIO] = Grid.MyInk.value
    grids[dark_grid_special_ratios > GRID_PIXEL_RATIO] = Grid.MySpecial.value
    costs = np.sum(cost_ratios > GRID_PIXEL_RATIO, axis=1)

    if debug:
        img2 = img.copy()
        grid_rois = np.array([util.numpy_to_opencv(idx) for grid in DECK_GRID_NUMPY_ROI_TOP_LEFTS for idx in grid]).reshape((15, 64, 2))
        cost_rois = np.array([util.numpy_to_opencv(idx) for grid in DECK_COST_NUMPY_ROI_TOP_LEFTS for idx in grid]).reshape((15, 6, 2))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ink_mask = cv2.inRange(hsv, ink_lower_bound, ink_upper_bound)
        special_mask = cv2.inRange(hsv, special_lower_bound, special_upper_bound)
        dark_ink_mask = cv2.inRange(hsv, MY_INK_DARKER_COLOR_HSV_LOWER_BOUND, MY_INK_DARKER_COLOR_HSV_UPPER_BOUND)
        dark_special_mask = cv2.inRange(hsv, MY_SPECIAL_DARKER_COLOR_HSV_LOWER_BOUND, MY_SPECIAL_DARKER_COLOR_HSV_UPPER_BOUND)
        # mask = np.maximum(ink_mask, special_mask)
        mask = np.max([ink_mask, special_mask, dark_ink_mask, dark_special_mask], axis=0)
        mask = cv2.merge((mask, mask, mask))
        for i, grid in enumerate(grid_rois):
            for k, roi in enumerate(grid):
                cv2.rectangle(img2, roi, roi + (DECK_GRID_ROI_WIDTH, DECK_GRID_ROI_HEIGHT), (0, 255, 0), 1)
                cv2.rectangle(mask, roi, roi + (DECK_GRID_ROI_WIDTH, DECK_GRID_ROI_HEIGHT), (0, 255, 0), 1)
                if grids[i][k] == Grid.MyInk.value:
                    cv2.rectangle(img2, roi, roi + (DECK_GRID_ROI_WIDTH, DECK_GRID_ROI_HEIGHT), (0, 0, 255), 1)
                    cv2.rectangle(mask, roi, roi + (DECK_GRID_ROI_WIDTH, DECK_GRID_ROI_HEIGHT), (0, 0, 255), 1)
                if grids[i][k] == Grid.MySpecial.value:
                    cv2.rectangle(img2, roi, roi + (DECK_GRID_ROI_WIDTH, DECK_GRID_ROI_HEIGHT), (255, 0, 0), 1)
                    cv2.rectangle(mask, roi, roi + (DECK_GRID_ROI_WIDTH, DECK_GRID_ROI_HEIGHT), (255, 0, 0), 1)
        for i, cost in enumerate(cost_rois):
            for k, roi in enumerate(cost):
                cv2.rectangle(img2, roi, roi + (DECK_COST_ROI_WIDTH, DECK_COST_ROI_HEIGHT), (0, 255, 0), 1)
                cv2.rectangle(mask, roi, roi + (DECK_COST_ROI_WIDTH, DECK_COST_ROI_HEIGHT), (0, 255, 0), 1)
                if cost_ratios[i][k] > GRID_PIXEL_RATIO:
                    cv2.rectangle(img2, roi, roi + (DECK_COST_ROI_WIDTH, DECK_COST_ROI_HEIGHT), (255, 0, 0), 1)
                    cv2.rectangle(mask, roi, roi + (DECK_COST_ROI_WIDTH, DECK_COST_ROI_HEIGHT), (255, 0, 0), 1)
            cv2.putText(mask, f'{costs[i]}', DECK_COST_OPENCV_ROI_LEFT_TOP[i] + np.rint([-DECK_GRID_ROI_WIDTH * 1.5, DECK_COST_ROI_HEIGHT]).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        debug.show('deck.image', img2)
        debug.show('deck.color_mask', mask)
    grids = grids.reshape((15, 8, 8))
    deck = [Card(grids[i], costs[i]) for i in range(15)]
    logger.debug(f'detection.deck: return={deck}')
    return deck
