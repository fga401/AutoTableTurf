from typing import Optional, Tuple

import cv2
import numpy as np

from logger import logger
from tableturf.debugger.interface import Debugger
from tableturf.manager.detection import util
from tableturf.manager.detection.ui import special_on
from tableturf.model import Stage, Pattern, Grid

BOUNDING_BOX_TOP_LEFT = np.array([0, 750])
BOUNDING_BOX_WIDTH = 850
BOUNDING_BOX_HEIGHT = 1080

CC_WIDTH_UPPER_BOUND = 45
CC_WIDTH_LOWER_BOUND = 30
CC_HEIGHT_UPPER_BOUND = 45
CC_HEIGHT_LOWER_BOUND = 30
CC_AREA_UPPER_BOUND = CC_WIDTH_UPPER_BOUND * CC_HEIGHT_UPPER_BOUND
CC_AREA_LOWER_BOUND = CC_WIDTH_LOWER_BOUND * CC_HEIGHT_LOWER_BOUND

EMPTY_COLOR_HSV_UPPER_BOUND = (255, 255, 46)
EMPTY_COLOR_HSV_LOWER_BOUND = (0, 0, 0)
ROI_EROSION_SIZE = 6


def _classify_connected_components(stats):
    """
    :param stats: ndarray of (x, y, width, height, area). shape = (N, 5)
    :return: ndarray of bool indicating target components. shape = (N,)
    """
    stats = stats[:, 2:5]
    upper_bound = np.array([CC_WIDTH_UPPER_BOUND, CC_HEIGHT_UPPER_BOUND, CC_AREA_UPPER_BOUND])[np.newaxis, ...]
    lower_bound = np.array([CC_WIDTH_LOWER_BOUND, CC_HEIGHT_LOWER_BOUND, CC_AREA_LOWER_BOUND])[np.newaxis, ...]
    return np.bitwise_and(np.all(stats > lower_bound, axis=1), np.all(stats < upper_bound, axis=1))


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def _get_steps(centers, width, height):
    """
    calculate the mean distant of all pairs to get the steps
    """
    idx = np.arange(centers.shape[0])
    pairs = cartesian_product(idx, idx)
    width_dist = np.abs(centers[pairs[:, 0]] - centers[pairs[:, 1]])[:, 0]
    height_dist = np.abs(centers[pairs[:, 0]] - centers[pairs[:, 1]])[:, 1]
    # filter out those pairs that is in the same row/col
    width_idx = width_dist > (width / 2)
    height_idx = height_dist > (height / 2)
    width_step = np.mean(width_dist[width_idx] / np.round(width_dist / width)[width_idx])
    height_step = np.mean(height_dist[height_idx] / np.round(height_dist / height)[height_idx])
    return width_step, height_step


def _mean_axis(coordinates, step):
    y = np.sort(coordinates)
    y_sum = np.insert(np.cumsum(y), 0, 0)
    y_idx = np.insert(np.where(y[1:] - y[:-1] > step / 2)[0] + 1, 0, 0)
    if y_idx[-1] != y.size:
        y_idx = np.insert(y_idx, y_idx.size, y.size)
    y = (y_sum[y_idx[1:]] - y_sum[y_idx[:-1]]) / (y_idx[1:] - y_idx[:-1])  # group points by distance and use mean value as y
    x_diff = np.insert(np.rint((y[1:] - y[:-1]) / step), 0, 0)
    x = np.cumsum(x_diff).astype(int)
    return y, x


def _spawn_axis(x, xp, fp, threshold=3):
    result = np.zeros_like(x)
    if xp.size < threshold:
        return result
    less_than_left = x < np.min(xp)
    greater_than_right = x > np.max(xp)
    between_left_and_right = np.bitwise_and(np.bitwise_not(less_than_left), np.bitwise_not(greater_than_right))
    # left extrapolation
    if np.any(less_than_left):
        left = np.poly1d(np.polyfit(xp[:threshold], fp[:threshold], 1))
        result[less_than_left] = left(x[less_than_left])
    # right extrapolation
    if np.any(greater_than_right):
        right = np.poly1d(np.polyfit(xp[-threshold:], fp[-threshold:], 1))
        result[greater_than_right] = right(x[greater_than_right])
    # interpolation
    if np.any(between_left_and_right):
        result[between_left_and_right] = np.interp(x[between_left_and_right], xp, fp)
    return result


def _spawn_roi_centers(centers, width_step, height_step):
    """
    :param centers: (N, 2)
    :return: (nx, ny, 2)
    """
    x = centers[:, 0]
    y = centers[:, 1]
    x_mean, x_idx = _mean_axis(x, width_step)
    y_mean, y_idx = _mean_axis(y, height_step)
    # number of points calculated by left extrapolation
    lx = np.round(x_mean[0] / width_step).astype(int)
    rx = np.round((BOUNDING_BOX_WIDTH - x_mean[-1]) / width_step).astype(int)
    ly = np.round(y_mean[0] / height_step).astype(int)
    ry = np.round((BOUNDING_BOX_HEIGHT - y_mean[-1]) / height_step).astype(int)
    # number of roi
    nx = x_idx[-1] + 1 + lx + rx
    ny = y_idx[-1] + 1 + ly + ry
    print(lx, ly)
    print(rx, ry)
    print(nx, ny)
    grid = np.zeros((ny, nx, 2, 2))

    # fill known centers
    def __index(center):
        return np.array([y_idx[np.argmin(np.abs(y_mean - center[0]))], x_idx[np.argmin(np.abs(x_mean - center[1]))]]) + (ly, lx)

    centers = np.array([util.numpy_to_opencv(center) for center in centers])
    indices = np.array([__index(center) for center in centers])
    grid[indices[:, 0], indices[:, 1]] = centers[:, np.newaxis, :]

    for i in range(ly, ny - ry):
        existed = np.all(grid[i, :] != 0, axis=(1, 2))
        points = np.argwhere(existed).reshape(-1)
        values = grid[i, points].mean(axis=1)
        new_points = np.argwhere(np.bitwise_not(existed)).reshape(-1)
        grid[i, new_points, 1, 0] = _spawn_axis(new_points, points, values[:, 0])
        grid[i, new_points, 1, 1] = _spawn_axis(new_points, points, values[:, 1])
    for j in range(lx, nx - rx):
        existed = np.all(grid[:, j] != 0, axis=(1, 2))
        points = np.argwhere(existed).reshape(-1)
        values = grid[points, j].mean(axis=1)
        new_points = np.argwhere(np.bitwise_not(existed)).reshape(-1)
        grid[new_points, j, 0, 0] = _spawn_axis(new_points, points, values[:, 0])
        grid[new_points, j, 0, 1] = _spawn_axis(new_points, points, values[:, 1])

    # fill missing values
    hidx = np.all(grid[:, :, 0] == 0, axis=2)
    vidx = np.all(grid[:, :, 1] == 0, axis=2)
    if np.any(hidx):
        grid[np.stack([hidx, np.zeros_like(hidx)], axis=-1)] = grid[np.stack([np.zeros_like(hidx), hidx], axis=-1)]
    if np.any(vidx):
        grid[np.stack([np.zeros_like(hidx), vidx], axis=-1)] = grid[np.stack([vidx, np.zeros_like(hidx)], axis=-1)]

    idx = np.bitwise_and(hidx, vidx)
    for i in np.argwhere(np.any(idx, axis=1)).reshape(-1):
        existed = np.all(grid[i, :] != 0, axis=(1, 2))
        points = np.argwhere(existed).reshape(-1)
        values = grid[i, points].mean(axis=1)
        new_points = np.argwhere(np.bitwise_not(existed)).reshape(-1)
        grid[i, new_points, 1, 0] = _spawn_axis(new_points, points, values[:, 0])
        grid[i, new_points, 1, 1] = _spawn_axis(new_points, points, values[:, 1])
    for j in np.argwhere(np.any(idx, axis=0)).reshape(-1):
        existed = np.all(grid[:, j] != 0, axis=(1, 2))
        points = np.argwhere(existed).reshape(-1)
        values = grid[points, j].mean(axis=1)
        new_points = np.argwhere(np.bitwise_not(existed)).reshape(-1)
        grid[new_points, j, 0, 0] = _spawn_axis(new_points, points, values[:, 0])
        grid[new_points, j, 0, 1] = _spawn_axis(new_points, points, values[:, 1])

    return np.rint(grid.mean(axis=2)).astype(int) + BOUNDING_BOX_TOP_LEFT


def stage_rois(img: np.ndarray, debug: Optional[Debugger] = None) -> (np.ndarray, int, int):
    """
    :return: top_lefts, roi_width, roi_height
    """
    # find connected components
    bounding_box = img[BOUNDING_BOX_TOP_LEFT[0]: BOUNDING_BOX_TOP_LEFT[0] + BOUNDING_BOX_HEIGHT, BOUNDING_BOX_TOP_LEFT[1]:BOUNDING_BOX_TOP_LEFT[1] + BOUNDING_BOX_WIDTH]
    hsv = cv2.cvtColor(bounding_box, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, EMPTY_COLOR_HSV_LOWER_BOUND, EMPTY_COLOR_HSV_UPPER_BOUND)
    mask = cv2.erode(mask, kernel=None)
    num_labels, roi_labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    # list all rois from stats
    roi_index = _classify_connected_components(stats)
    square_width, square_height = stats[roi_index, 2:4].mean(axis=0)
    roi_width_step, roi_height_step = _get_steps(centroids[roi_index], square_width, square_height)
    roi_centers = _spawn_roi_centers(centroids[roi_index], roi_width_step, roi_height_step)
    roi_width, roi_height = np.round([roi_width_step, roi_height_step]).astype(int) - ROI_EROSION_SIZE
    rois = roi_centers - np.rint([roi_width / 2, roi_height / 2])[np.newaxis, ...].astype(int)
    # trim rois
    row_mask = np.bitwise_and(np.all(rois[:, :, 0] >= 0, axis=1), np.all(rois[:, :, 0] + roi_height < BOUNDING_BOX_TOP_LEFT[0] + BOUNDING_BOX_HEIGHT, axis=1))
    col_mask = np.bitwise_and(np.all(rois[:, :, 1] >= 0, axis=0), np.all(rois[:, :, 1] + roi_width < BOUNDING_BOX_TOP_LEFT[1] + BOUNDING_BOX_WIDTH, axis=0))
    roi_centers = roi_centers[row_mask][:, col_mask]
    rois = rois[row_mask][:, col_mask]

    stage_grid = stage(img, rois, roi_width, roi_height, last_stage=None, debug=debug)[0].grid
    not_wall = stage_grid != Grid.Wall.value
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(not_wall.astype(np.uint8) * 255)
    target_cc = np.argmax(stats[1:, 4]) + 1
    not_wall[labels != target_cc] = False
    row_mask = np.any(not_wall, axis=1)
    col_mask = np.any(not_wall, axis=0)

    def __expand_mask(mask: np.ndarray) -> np.ndarray:
        idx = np.argwhere(mask)
        if idx.size == 0:
            return mask
        lower = np.maximum(0, np.min(idx) - 1)
        upper = np.minimum(mask.size - 1, np.max(idx) + 1)
        mask[lower] = True
        mask[upper] = True
        return mask

    row_mask = __expand_mask(row_mask)
    col_mask = __expand_mask(col_mask)
    roi_centers = roi_centers[row_mask][:, col_mask]
    rois = rois[row_mask][:, col_mask]

    if debug:
        _rois = rois.reshape(-1, 2)
        roi_centers = roi_centers.reshape(-1, 2)
        colorful_mask = cv2.merge((roi_labels * 53 % 255, roi_labels * 101 % 255, roi_labels * 151 % 255))
        img_mask = np.zeros_like(img)
        for center in centroids[roi_index]:
            cv2.circle(colorful_mask, np.rint(center).astype(int), 2, (0, 255, 0), 3)
        img_mask[BOUNDING_BOX_TOP_LEFT[0]: BOUNDING_BOX_TOP_LEFT[0] + BOUNDING_BOX_HEIGHT, BOUNDING_BOX_TOP_LEFT[1]:BOUNDING_BOX_TOP_LEFT[1] + BOUNDING_BOX_WIDTH] = colorful_mask
        for center in roi_centers:
            center = util.numpy_to_opencv(center)
            cv2.circle(img_mask, np.rint(center).astype(int), 2, (0, 0, 255), 3)
        for roi in _rois:
            roi = util.numpy_to_opencv(roi)
            cv2.rectangle(img_mask, roi, roi + (roi_width, roi_height), (255, 255, 255), 1)
        debug.show('stage_rois.color_mask', colorful_mask)
        debug.show('stage_rois.edge_mask', img_mask)

    logger.debug(f'detection.stage_rois: return={rois, roi_width, roi_height}')
    return rois, roi_width, roi_height


COLOR_PIXEL_RATIO = 0.1
# hsv color range
NEUTRAL_COLOR_HSV_UPPER_BOUND = (255, 50, 220)
NEUTRAL_COLOR_HSV_LOWER_BOUND = (0, 0, 180)
MY_INK_COLOR_HSV_UPPER_BOUND = (40, 255, 255)
MY_INK_COLOR_HSV_LOWER_BOUND = (25, 220, 220)
MY_SPECIAL_COLOR_HSV_UPPER_BOUND = (25, 255, 255)
MY_SPECIAL_COLOR_HSV_LOWER_BOUND = (10, 220, 220)
HIS_INK_COLOR_HSV_UPPER_BOUND = (120, 190, 255)
HIS_INK_COLOR_HSV_LOWER_BOUND = (110, 170, 220)
HIS_SPECIAL_COLOR_HSV_UPPER_BOUND = (100, 255, 255)
HIS_SPECIAL_COLOR_HSV_LOWER_BOUND = (80, 80, 220)
HIS_FIERY_SPECIAL_COLOR_HSV_UPPER_BOUND = (100, 30, 255)
HIS_FIERY_SPECIAL_COLOR_HSV_LOWER_BOUND = (80, 0, 230)
STAGE_SQUARE_CANNY_THRESHOLD = (20, 60)
MY_INK_TOP_COLOR_HSV_UPPER_BOUND = (40, 140, 255)
MY_INK_TOP_COLOR_HSV_LOWER_BOUND = (30, 80, 220)
MY_SPECIAL_TOP_COLOR_HSV_UPPER_BOUND = (30, 255, 255)
MY_SPECIAL_TOP_COLOR_HSV_LOWER_BOUND = (20, 150, 220)
DEBUG_COLOR = {
    Grid.Empty.value: (255, 255, 255),
    Grid.MyInk.value: (0, 255, 255),
    Grid.MySpecial.value: (0, 191, 255),
    Grid.HisInk.value: (255, 0, 0),
    Grid.HisSpecial.value: (255, 0, 128),
    Grid.Neutral.value: (127, 127, 127),
    Grid.Wall.value: (0, 255, 0),
}


def stage(img: np.ndarray, rois: np.ndarray, roi_width, roi_height, last_stage: Optional[Stage] = None, debug: Optional[Debugger] = None) -> (Stage, np.ndarray):
    """
    :param rois: (h, w, 2), rois[i][j] = (y, x)
    :return: stage and is_fiery
    """
    h, w, _ = rois.shape
    rois = rois.reshape((h * w, 2))
    roi_size = roi_width * roi_height

    def __roi(top_left: np.ndarray):
        return img[top_left[0]:top_left[0] + roi_height, top_left[1]:top_left[1] + roi_width]

    def __square(k, top_left: np.ndarray):
        roi = img[top_left[0]:top_left[0] + roi_height, top_left[1]:top_left[1] + roi_width]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        background = cv2.inRange(hsv, EMPTY_COLOR_HSV_LOWER_BOUND, EMPTY_COLOR_HSV_UPPER_BOUND)
        background_ratio = np.sum(background == 255) / roi_size
        if background_ratio > COLOR_PIXEL_RATIO:
            return Grid.Empty.value, 0
        neutral = cv2.inRange(hsv, NEUTRAL_COLOR_HSV_LOWER_BOUND, NEUTRAL_COLOR_HSV_UPPER_BOUND)
        neutral_ratio = np.sum(neutral == 255) / roi_size
        if neutral_ratio > COLOR_PIXEL_RATIO:
            return Grid.Neutral.value, 0
        his_ink = cv2.inRange(hsv, HIS_INK_COLOR_HSV_LOWER_BOUND, HIS_INK_COLOR_HSV_UPPER_BOUND)
        his_ink_ratio = np.sum(his_ink == 255) / roi_size
        if his_ink_ratio > COLOR_PIXEL_RATIO:
            return Grid.HisInk.value, 0

        sub_hsv = hsv[roi_height // 5 * 2:roi_height // 5 * 3, roi_width // 2:roi_width // 5 * 3]
        his_fiery_sp = cv2.inRange(sub_hsv, HIS_FIERY_SPECIAL_COLOR_HSV_LOWER_BOUND, HIS_FIERY_SPECIAL_COLOR_HSV_UPPER_BOUND)
        his_fiery_sp_ratio = np.mean(his_fiery_sp == 255)
        if his_fiery_sp_ratio > COLOR_PIXEL_RATIO:
            return Grid.HisSpecial.value, 1
        his_sp = cv2.inRange(sub_hsv, HIS_SPECIAL_COLOR_HSV_LOWER_BOUND, HIS_SPECIAL_COLOR_HSV_UPPER_BOUND)
        his_sp_ratio = np.mean(his_sp == 255)
        if his_sp_ratio > COLOR_PIXEL_RATIO:
            return Grid.HisSpecial.value, 0
        my_sp = cv2.inRange(sub_hsv, MY_SPECIAL_COLOR_HSV_LOWER_BOUND, MY_SPECIAL_COLOR_HSV_UPPER_BOUND)
        my_sp_ratio = np.mean(my_sp == 255)
        if my_sp_ratio > COLOR_PIXEL_RATIO:
            return Grid.MySpecial.value, 0

        my_ink = cv2.inRange(sub_hsv, MY_INK_COLOR_HSV_LOWER_BOUND, MY_INK_COLOR_HSV_UPPER_BOUND)
        my_ink_ratio = np.mean(my_ink == 255)
        if my_ink_ratio > COLOR_PIXEL_RATIO:
            edge = cv2.Canny(__roi(top_left), *STAGE_SQUARE_CANNY_THRESHOLD)
            edge = cv2.dilate(edge, kernel=np.ones((3, 3), dtype=np.uint8))
            edge = edge + 1  # inverse background and foreground
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edge)
            sub_roi_idx = np.argwhere(centroids[:, 1] <= roi_height // 4)
            sub_roi_idx = np.isin(labels, sub_roi_idx)
            sub_hsv = hsv[sub_roi_idx][np.newaxis, ...]
            if sub_hsv.size == 0:
                return Grid.MyInk.value, 0
            ink_size = np.sum(cv2.inRange(sub_hsv, MY_INK_TOP_COLOR_HSV_LOWER_BOUND, MY_INK_TOP_COLOR_HSV_UPPER_BOUND) == 255)
            sp_size = np.sum(cv2.inRange(sub_hsv, MY_SPECIAL_TOP_COLOR_HSV_LOWER_BOUND, MY_SPECIAL_TOP_COLOR_HSV_UPPER_BOUND) == 255)
            if sp_size >= ink_size:
                return Grid.MySpecial.value, 1
            else:
                return Grid.MyInk.value, 0
        return Grid.Wall.value, 0

    stage = np.array([__square(k, top_left) for k, top_left in enumerate(rois)])
    is_fiery = stage[:, 1]
    stage = stage[:, 0]
    if last_stage is not None:
        last_stage_grid = last_stage
        stage[last_stage_grid == Grid.Neutral.value] = Grid.Neutral.value
        stage[last_stage_grid == Grid.MySpecial.value] = Grid.MySpecial.value
        stage[last_stage_grid == Grid.HisSpecial.value] = Grid.HisSpecial.value

    if debug:
        img2 = img.copy()
        opencv_rois = np.array([util.numpy_to_opencv(idx) for idx in rois])
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # mask_color = cv2.inRange(hsv, HIS_SPECIAL_COLOR_HSV_LOWER_BOUND, HIS_SPECIAL_COLOR_HSV_UPPER_BOUND)
        mask_edge = np.zeros_like(img)
        # mask_color = cv2.merge([mask_color, mask_color, mask_color])
        for k, left_top in enumerate(opencv_rois):
            roi = __roi(rois[k])
            _hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            edge = cv2.Canny(roi, 20, 60)
            edge = cv2.dilate(edge, kernel=np.ones((2, 2), dtype=np.uint8))
            edge = edge + 1  # inverse background and foreground
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, connectivity=8)
            color = DEBUG_COLOR[stage[k]]
            if is_fiery[k]:
                thickness = 2
            else:
                thickness = 1
            mask_edge[rois[k][0]:rois[k][0] + roi_height, rois[k][1]:rois[k][1] + roi_width] = (labels * 47 % 255)[..., np.newaxis]
            cv2.rectangle(img2, left_top, left_top + (roi_width, roi_height), color, thickness)
            cv2.putText(img2, f'{k}', left_top + np.rint([roi_width / 10, roi_height / 1.3]).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            cv2.rectangle(mask_edge, left_top, left_top + (roi_width, roi_height), color, thickness)
            cv2.putText(mask_edge, f'{k}', left_top + np.rint([roi_width / 10, roi_height / 1.3]).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        debug.show('stage.image', img2)
        debug.show('stage.edge_mask', mask_edge)
        # debug.show('color_mask', mask_color)

    stage = stage.reshape((h, w))
    is_fiery = is_fiery.reshape((h, w))
    # mark all invalid fiery special squares as trivial squares
    valid_fiery_sp = Stage(stage).my_fiery_sp
    all_fiery_sp = np.argwhere(np.bitwise_and(is_fiery, stage == Grid.MySpecial.value))
    invalid_fiery_sp = all_fiery_sp[np.bitwise_not(np.array([idx in valid_fiery_sp for idx in all_fiery_sp], dtype=bool))]
    is_fiery[invalid_fiery_sp] = False
    stage[invalid_fiery_sp] = Grid.MyInk.value
    stage = Stage(stage)

    logger.debug(f'detection.stage: return={stage, is_fiery}')
    return stage, is_fiery


PREVIEW_SQUARE_CANNY_THRESHOLD = (50, 80)
PREVIEW_INK_DIAGONAL_RATIO = 0.8
PREVIEW_INK_RATIO = 0.2

PREVIEW_SPECIAL_RATIO = 0.025
PREVIEW_MY_FIRE_COLOR_HSV_RANGES = [
    [(20, 125, 240), (30, 140, 255)]
]
PREVIEW_HIS_FIRE_COLOR_HSV_RANGES = [
    [(80, 25, 235), (90, 45, 245)],
]
PREVIEW_EMPTY_COLOR_HSV_RANGES = [
    [(0, 0, 180), (255, 50, 220)],
]
PREVIEW_EMPTY_ORANGE_COLOR_HSV_RANGES = [
    [(10, 130, 160), (20, 255, 220)],
]
PREVIEW_NEUTRAL_COLOR_HSV_RANGES = [
    [(0, 0, 205), (120, 15, 225)]
]
PREVIEW_MY_INK_COLOR_HSV_RANGES = [
    [(25, 90, 230), (35, 190, 240)],
]
PREVIEW_HIS_INK_COLOR_HSV_RANGES = [
    [(110, 70, 240), (120, 100, 255)],
]
PREVIEW_MY_SPECIAL_COLOR_HSV_RANGES = [
    [(10, 95, 240), (20, 130, 255)],
]
PREVIEW_HIS_SPECIAL_COLOR_HSV_RANGES = [
    [(90, 100, 240), (100, 130, 255)],
]
PREVIEW_MY_FIERY_SPECIAL_COLOR_HSV_RANGES = [
    [(20, 120, 240), (30, 140, 250)],
    [(15, 110, 230), (25, 150, 255)],
    [(20, 125, 240), (30, 140, 255)],
]
PREVIEW_HIS_FIERY_SPECIAL_COLOR_HSV_RANGES = [
    [(80, 10, 230), (90, 15, 245)],
    [(90, 70, 235), (100, 90, 245)],
    [(80, 25, 235), (90, 45, 245)],
]
PREVIEW_WALL_COLOR_HSV_RANGES = [
    [(120, 10, 150), (130, 80, 255)],
]

PREVIEW_MY_INK_DARKER_GRAY_COLOR_HSV_RANGES = [
    [(20, 50, 170), (40, 80, 220)],
]
PREVIEW_HIS_INK_DARKER_GRAY_COLOR_HSV_RANGES = [
    [(110, 50, 180), (120, 80, 240)],
]
PREVIEW_MY_INK_DARKER_ORANGE_COLOR_HSV_RANGES = [
    [(20, 180, 190), (30, 250, 210)],
]
PREVIEW_HIS_INK_DARKER_ORANGE_COLOR_HSV_RANGES = [
    [(0, 40, 180), (10, 70, 190)],
    [(140, 45, 120), (180, 80, 190)],
]


def preview(img: np.ndarray, stage: Stage, is_fiery: np.ndarray, rois: np.ndarray, roi_width, roi_height, debug: Optional[Debugger] = None) -> Tuple[Optional[Pattern], Optional[np.ndarray]]:
    h, w, _ = rois.shape
    stage_grid = stage.grid.reshape((h * w))
    is_fiery = is_fiery.reshape((h * w))
    rois = rois.reshape((h * w, 2))
    sp_on = special_on(img)

    def __roi(top_left: np.ndarray):
        return img[top_left[0]:top_left[0] + roi_height, top_left[1]:top_left[1] + roi_width]

    def __is_ink_preview(k, roi: np.ndarray) -> bool:
        edge = cv2.Canny(roi, *PREVIEW_SQUARE_CANNY_THRESHOLD)
        edge = cv2.dilate(edge, kernel=np.ones((2, 2), dtype=np.uint8))
        edge = np.rot90(edge)
        is_diagonal = [np.mean(np.diag(edge, i) == 255) > PREVIEW_INK_DIAGONAL_RATIO for i in range(-roi_width // 2, roi_width // 2)]
        return np.mean(is_diagonal) > PREVIEW_INK_RATIO

    def __in_ranges(roi, ranges):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        return np.any([cv2.inRange(hsv, lower_bound, upper_bound) for lower_bound, upper_bound in ranges], axis=0)

    def __is_valid_mask(mask, min_area=4, min_num=3) -> bool:
        if np.mean(mask) < PREVIEW_SPECIAL_RATIO:
            return False
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        ares = stats[1:, 4]
        valid = np.bitwise_and(ares >= min_area, ares < 50)
        if np.sum(valid) < min_num:
            return False
        return True

    def __is_special_preview(k, roi: np.ndarray) -> bool:
        extra_ranges = []
        if k + w < h * w and is_fiery[k + w]:
            if stage_grid[k + w] == Grid.MySpecial.value:
                extra_ranges = PREVIEW_MY_FIRE_COLOR_HSV_RANGES
            elif stage_grid[k + w] == Grid.HisSpecial.value:
                extra_ranges = PREVIEW_HIS_FIRE_COLOR_HSV_RANGES
        if stage_grid[k] == Grid.Empty.value:
            mask = __in_ranges(roi, PREVIEW_EMPTY_COLOR_HSV_RANGES + extra_ranges)
            if __is_valid_mask(mask):
                return True
            mask = __in_ranges(roi, PREVIEW_EMPTY_ORANGE_COLOR_HSV_RANGES + extra_ranges)
            if __is_valid_mask(mask, min_area=6, min_num=5):
                return True
        if stage_grid[k] == Grid.Neutral.value:
            mask = __in_ranges(roi, PREVIEW_NEUTRAL_COLOR_HSV_RANGES + extra_ranges)
            if __is_valid_mask(mask):
                return True
        if stage_grid[k] == Grid.MyInk.value:
            if not sp_on:
                mask = __in_ranges(roi, PREVIEW_MY_INK_COLOR_HSV_RANGES + extra_ranges)
                if __is_valid_mask(mask):
                    return True
            else:
                mask = __in_ranges(roi, PREVIEW_MY_INK_DARKER_GRAY_COLOR_HSV_RANGES + extra_ranges)
                if __is_valid_mask(mask):
                    return True
                mask = __in_ranges(roi, PREVIEW_MY_INK_DARKER_ORANGE_COLOR_HSV_RANGES + extra_ranges)
                if __is_valid_mask(mask, min_area=6, min_num=4):
                    return True
        if stage_grid[k] == Grid.HisInk.value:
            if not sp_on:
                mask = __in_ranges(roi, PREVIEW_HIS_INK_COLOR_HSV_RANGES + extra_ranges)
                if __is_valid_mask(mask):
                    return True
            else:
                mask = __in_ranges(roi, PREVIEW_HIS_INK_DARKER_GRAY_COLOR_HSV_RANGES + extra_ranges)
                if __is_valid_mask(mask):
                    return True
                mask = __in_ranges(roi, PREVIEW_HIS_INK_DARKER_ORANGE_COLOR_HSV_RANGES + extra_ranges)
                if __is_valid_mask(mask, min_area=6, min_num=4):
                    return True
        if stage_grid[k] == Grid.MySpecial.value:
            if is_fiery[k]:
                mask = __in_ranges(roi, PREVIEW_MY_FIERY_SPECIAL_COLOR_HSV_RANGES + extra_ranges)
            else:
                mask = __in_ranges(roi, PREVIEW_MY_SPECIAL_COLOR_HSV_RANGES + extra_ranges)
            if __is_valid_mask(mask):
                return True
        if stage_grid[k] == Grid.HisSpecial.value:
            if is_fiery[k]:
                mask = __in_ranges(roi, PREVIEW_HIS_FIERY_SPECIAL_COLOR_HSV_RANGES + extra_ranges)
            else:
                mask = __in_ranges(roi, PREVIEW_HIS_SPECIAL_COLOR_HSV_RANGES + extra_ranges)
            if __is_valid_mask(mask):
                return True
        if stage_grid[k] == Grid.Wall.value:
            mask = __in_ranges(roi, PREVIEW_WALL_COLOR_HSV_RANGES + extra_ranges)
            if __is_valid_mask(mask):
                return True
        return False

    def __square(k, top_left: np.ndarray):
        roi = img[top_left[0]:top_left[0] + roi_height, top_left[1]:top_left[1] + roi_width]
        if __is_ink_preview(k, roi):
            return Grid.MyInk.value
        if __is_special_preview(k, roi):
            return Grid.MySpecial.value
        return Grid.Empty.value

    pattern = np.array([__square(k, top_left) for k, top_left in enumerate(rois)])
    pattern = pattern.reshape((h, w))
    no_pattern = np.all(pattern == Grid.Empty.value)
    if not no_pattern:
        pattern_mask = (pattern != Grid.Empty.value).astype(np.uint8)
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(pattern_mask, connectivity=8)
        target_label = np.argmax(stats[1:, 4]) + 1
        max_size = stats[target_label, 4]
        if max_size == 1:
            indexes = centroids[1:, 1] * w + centroids[1:, 0]
            in_wall = stage_grid[indexes] == Grid.Wall.value
            if np.all(in_wall):
                dist = np.power(centroids[1:] - (w // 2, h // 2), 2)
                target_label = np.argmin(dist) + 1
            else:
                target_label = np.argwhere(np.bitwise_not(in_wall)) + 1
        pattern[np.bitwise_not(labels == target_label)] = Grid.Empty.value

    if debug:
        img2 = img.copy()
        _pattern = pattern.reshape(-1)
        opencv_rois = np.array([util.numpy_to_opencv(idx) for idx in rois])
        mask_edge = np.zeros_like(img)
        mask_color = __in_ranges(img, PREVIEW_MY_INK_DARKER_ORANGE_COLOR_HSV_RANGES).astype(np.uint8) * 255
        # calculate
        mask_color = cv2.merge([mask_color, mask_color, mask_color])
        for k, left_top in enumerate(opencv_rois):
            roi = __roi(rois[k])
            _hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            _gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _hsv[:, :, 0] = 0
            _hsv[:, :, 2] = 0
            color = DEBUG_COLOR[_pattern[k]]
            edge = cv2.Canny(_gray, 20, 30)
            # edge = cv2.dilate(edge, kernel=np.ones((2, 2), dtype=np.uint8))

            # draw
            mask_edge[rois[k][0]:rois[k][0] + roi_height, rois[k][1]:rois[k][1] + roi_width] = edge[..., np.newaxis]
            cv2.rectangle(img2, left_top, left_top + (roi_width, roi_height), color, 1)
            cv2.putText(img2, f'{k}', left_top + np.rint([roi_width / 10, roi_height / 1.3]).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            cv2.rectangle(mask_edge, left_top, left_top + (roi_width, roi_height), color, 1)
            cv2.putText(mask_edge, f'{k}', left_top + np.rint([roi_width / 10, roi_height / 1.3]).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            cv2.rectangle(mask_color, left_top, left_top + (roi_width, roi_height), color, 1)
            cv2.putText(mask_color, f'{k}', left_top + np.rint([roi_width / 10, roi_height / 1.3]).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        debug.show('preview.image', img2)
        debug.show('preview.edge_mask', mask_edge)
        debug.show('preview.color_mask', mask_color)
    if no_pattern:
        logger.debug(f'detection.preview: return=None')
        return None, None
    pattern = pattern.reshape((h, w))
    index = np.argwhere(pattern != Grid.Empty.value)[0]
    pattern = Pattern(pattern.reshape((h, w)))
    logger.debug(f'detection.preview: return={pattern, index}')
    return pattern, index


MY_SP_ROI_TOP_LEFTS = [(987, 61), (987, 231), (987, 399)]
HIS_SP_ROI_TOP_LEFTS = [(64, 50), (64, 183), (64, 314)]
SP_ROI_WIDTH = 15
SP_ROI_HEIGHT = 15
MY_SP_ROI_WIDTH_STEP = 32
HIS_SP_ROI_WIDTH_STEP = 25
SP_PIXEL_RATIO = 0.4


def sp(img: np.ndarray, debug: Optional[Debugger] = None) -> Tuple[int, int]:
    my_rois = np.concatenate([util.grid_roi_top_lefts(toe_left, width=5, height=1, width_step=MY_SP_ROI_WIDTH_STEP, height_step=0, width_offset=0, height_offset=0).squeeze() for toe_left in MY_SP_ROI_TOP_LEFTS])
    his_rois = np.concatenate([util.grid_roi_top_lefts(toe_left, width=5, height=1, width_step=HIS_SP_ROI_WIDTH_STEP, height_step=0, width_offset=0, height_offset=0).squeeze() for toe_left in HIS_SP_ROI_TOP_LEFTS])

    def __ratio(top_left: np.ndarray, lower_bound, upper_bound) -> float:
        roi = img[top_left[0]:top_left[0] + SP_ROI_HEIGHT, top_left[1]:top_left[1] + SP_ROI_WIDTH]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        return np.sum(mask == 255) / (SP_ROI_HEIGHT * SP_ROI_WIDTH)

    my_ratios = np.array([__ratio(top_left, MY_SPECIAL_COLOR_HSV_LOWER_BOUND, MY_SPECIAL_COLOR_HSV_UPPER_BOUND) for top_left in my_rois])
    his_ratios = np.array([__ratio(top_left, HIS_SPECIAL_COLOR_HSV_LOWER_BOUND, HIS_SPECIAL_COLOR_HSV_UPPER_BOUND) for top_left in his_rois])
    my_sp = np.sum(my_ratios > SP_PIXEL_RATIO)
    his_sp = np.sum(his_ratios > SP_PIXEL_RATIO)

    if debug:
        img2 = img.copy()
        hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        my_mask = cv2.inRange(hsv, MY_SPECIAL_COLOR_HSV_LOWER_BOUND, MY_SPECIAL_COLOR_HSV_UPPER_BOUND)
        his_mask = cv2.inRange(hsv, HIS_SPECIAL_COLOR_HSV_LOWER_BOUND, HIS_SPECIAL_COLOR_HSV_UPPER_BOUND)
        mask = np.maximum(my_mask, his_mask)
        mask = cv2.merge([mask, mask, mask])
        opencv_rois = np.array([util.numpy_to_opencv(idx) for idx in my_rois])
        for k, left_top in enumerate(opencv_rois):
            cv2.rectangle(img2, left_top, left_top + (SP_ROI_WIDTH, SP_ROI_HEIGHT), (0, 255, 0), 1)
            cv2.rectangle(mask, left_top, left_top + (SP_ROI_WIDTH, SP_ROI_HEIGHT), (0, 255, 0), 1)
        opencv_rois = np.array([util.numpy_to_opencv(idx) for idx in his_rois])
        for k, left_top in enumerate(opencv_rois):
            cv2.rectangle(img2, left_top, left_top + (SP_ROI_WIDTH, SP_ROI_HEIGHT), (0, 255, 0), 1)
            cv2.rectangle(mask, left_top, left_top + (SP_ROI_WIDTH, SP_ROI_HEIGHT), (0, 255, 0), 1)
        debug.show('sp.image', img2)
        debug.show('sp.color_mask', mask)
    logger.debug(f'detection.sp: return={my_sp, his_sp}')
    return my_sp, his_sp
