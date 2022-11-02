from typing import Iterable

import cv2
import numpy as np

from logger import logger
from tableturf.manager.detection import util
from tableturf.model import Stage, Pattern, Grid

BOUNDING_BOX_TOP_LEFT = np.array([0, 750])
BOUNDING_BOX_WIDTH = 800
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


def stage_rois(img: np.ndarray, debug=False) -> (np.ndarray, int, int):
    """
    :return: top_lefts, roi_width, roi_height
    """
    # find connected components
    bounding_box = img[BOUNDING_BOX_TOP_LEFT[0]: BOUNDING_BOX_TOP_LEFT[0] + BOUNDING_BOX_HEIGHT, BOUNDING_BOX_TOP_LEFT[1]:BOUNDING_BOX_TOP_LEFT[1] + BOUNDING_BOX_WIDTH]
    hsv = cv2.cvtColor(bounding_box, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, EMPTY_COLOR_HSV_LOWER_BOUND, EMPTY_COLOR_HSV_UPPER_BOUND)
    mask = cv2.erode(mask, kernel=None)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, connectivity=4)
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
    print(roi_width, roi_height)
    print(roi_width_step, roi_height_step)
    # TODO: trim col/row that is wall
    if debug:
        rois = rois.reshape(-1, 2)
        roi_centers = roi_centers.reshape(-1, 2)
        colorful_mask = cv2.merge((labels * 53 % 255, labels * 101 % 255, labels * 151 % 255))
        img_mask = np.zeros_like(img)
        for center in centroids[roi_index]:
            cv2.circle(colorful_mask, np.rint(center).astype(int), 2, (0, 255, 0), 3)
        img_mask[BOUNDING_BOX_TOP_LEFT[0]: BOUNDING_BOX_TOP_LEFT[0] + BOUNDING_BOX_HEIGHT, BOUNDING_BOX_TOP_LEFT[1]:BOUNDING_BOX_TOP_LEFT[1] + BOUNDING_BOX_WIDTH] = colorful_mask
        for center in roi_centers:
            center = util.numpy_to_opencv(center)
            cv2.circle(img_mask, np.rint(center).astype(int), 2, (0, 0, 255), 3)
        for roi in rois:
            roi = util.numpy_to_opencv(roi)
            cv2.rectangle(img_mask, roi, roi + (roi_width, roi_height), (255, 255, 255), 1)
        util.show(img)
        util.show(img_mask)
    return rois, roi_width, roi_height


COLOR_PIXEL_RATIO = 0.1
CIRCLE_PIXEL_RATIO = 0.75
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
SQUARE_COLOR_HSV_RANGES = np.array([
    (EMPTY_COLOR_HSV_LOWER_BOUND, EMPTY_COLOR_HSV_UPPER_BOUND),
    (NEUTRAL_COLOR_HSV_LOWER_BOUND, NEUTRAL_COLOR_HSV_UPPER_BOUND),
    (MY_INK_COLOR_HSV_LOWER_BOUND, MY_INK_COLOR_HSV_UPPER_BOUND),
    (MY_SPECIAL_COLOR_HSV_LOWER_BOUND, MY_SPECIAL_COLOR_HSV_UPPER_BOUND),
    (HIS_INK_COLOR_HSV_LOWER_BOUND, HIS_INK_COLOR_HSV_UPPER_BOUND),
    (HIS_SPECIAL_COLOR_HSV_LOWER_BOUND, HIS_SPECIAL_COLOR_HSV_UPPER_BOUND),
    (HIS_FIERY_SPECIAL_COLOR_HSV_LOWER_BOUND, HIS_FIERY_SPECIAL_COLOR_HSV_UPPER_BOUND),
])
SQUARE_GRID_INFO = [
    (Grid.MyInk.value, 0),
    (Grid.MySpecial.value, 0),
    (Grid.MySpecial.value, 1),
    (Grid.HisInk.value, 0),
    (Grid.HisSpecial.value, 0),
    (Grid.HisSpecial.value, 1),
    (Grid.Neutral.value, 0),
    (Grid.Empty.value, 0),
]

DEBUG_COLOR = {
    Grid.Empty.value: (255, 255, 255),
    Grid.MyInk.value: (0, 255, 255),
    Grid.MySpecial.value: (0, 191, 255),
    Grid.HisInk.value: (255, 0, 0),
    Grid.HisSpecial.value: (255, 0, 128),
    Grid.Neutral.value: (127, 127, 127),
    Grid.Wall.value: (0, 255, 0),
}


def stage(img: np.ndarray, rois: np.ndarray, roi_width, roi_height, debug=False) -> (Stage, Pattern):
    """
    :param rois: (h, w, 2), rois[i][j] = (y, x)
    :return: stage
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

        edge = cv2.Canny(__roi(top_left), 50, 100)
        edge = cv2.dilate(edge, kernel=np.ones((4, 4), dtype=np.uint8))
        edge = edge + 1  # inverse background and foreground
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, connectivity=8)
        sub_roi = labels[roi_height // 5 * 2:roi_height // 5 * 3, roi_width // 2:roi_width // 5 * 3]
        if np.all(sub_roi == 0):
            target = 0
        else:
            target = np.round(np.mean(sub_roi[sub_roi > 0])).astype(np.uint8)
        if target == 0:
            target = np.argmax(stats[1:, 4]) + 1
        sub_roi_idx = labels == target
        sub_hsv = hsv[sub_roi_idx][np.newaxis, ...]
        size = stats[target][4]

        his_fiery_sp = cv2.inRange(sub_hsv, HIS_FIERY_SPECIAL_COLOR_HSV_LOWER_BOUND, HIS_FIERY_SPECIAL_COLOR_HSV_UPPER_BOUND)
        his_fiery_sp_ratio = np.sum(his_fiery_sp == 255) / size
        if his_fiery_sp_ratio > COLOR_PIXEL_RATIO:
            return Grid.HisSpecial.value, 1
        his_sp = cv2.inRange(sub_hsv, HIS_SPECIAL_COLOR_HSV_LOWER_BOUND, HIS_SPECIAL_COLOR_HSV_UPPER_BOUND)
        his_sp_ratio = np.sum(his_sp == 255) / size
        if his_sp_ratio > COLOR_PIXEL_RATIO:
            return Grid.HisSpecial.value, 0
        my_sp = cv2.inRange(sub_hsv, MY_SPECIAL_COLOR_HSV_LOWER_BOUND, MY_SPECIAL_COLOR_HSV_UPPER_BOUND)
        my_sp_ratio = np.sum(my_sp == 255) / size
        if my_sp_ratio > COLOR_PIXEL_RATIO:
            return Grid.MySpecial.value, 0

        my_ink = cv2.inRange(sub_hsv, MY_INK_COLOR_HSV_LOWER_BOUND, MY_INK_COLOR_HSV_UPPER_BOUND)
        my_ink_ratio = np.sum(my_ink == 255) / size
        if my_ink_ratio > COLOR_PIXEL_RATIO:
            labels[np.bitwise_not(sub_roi_idx)] = 0
            labels = labels.astype(np.uint8)
            smooth_labels = cv2.dilate(labels, kernel=np.ones((3, 3), dtype=np.uint8))
            smooth_labels = cv2.erode(smooth_labels, kernel=np.ones((3, 3), dtype=np.uint8))
            area = np.sum(smooth_labels > 0)
            contours, hierarchy = cv2.findContours(labels, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour = contours[0]
            center, radius = cv2.minEnclosingCircle(contour)
            if area / (np.power(radius, 2) * np.pi) > CIRCLE_PIXEL_RATIO:
                return Grid.MySpecial.value, 1
            else:
                return Grid.MyInk.value, 0
        return Grid.Wall.value, 0

    stage = np.array([__square(k, top_left) for k, top_left in enumerate(rois)])
    is_fiery = stage[:, 1]
    stage = stage[:, 0]

    if debug:
        img_2 = img.copy()
        opencv_rois = np.array([util.numpy_to_opencv(idx) for idx in rois])
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # mask_color = cv2.inRange(hsv, HIS_SPECIAL_COLOR_HSV_LOWER_BOUND, HIS_SPECIAL_COLOR_HSV_UPPER_BOUND)
        mask_edge = np.zeros_like(img)
        # mask_color = cv2.merge([mask_color, mask_color, mask_color])
        for k, left_top in enumerate(opencv_rois):
            roi = __roi(rois[k])
            _hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            edge = cv2.Canny(roi, 50, 90)
            edge = cv2.dilate(edge, kernel=np.ones((3, 3), dtype=np.uint8))
            edge = edge + 1  # inverse background and foreground
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, connectivity=8)
            color = DEBUG_COLOR[stage[k]]
            mask_edge[rois[k][0]:rois[k][0] + roi_height, rois[k][1]:rois[k][1] + roi_width] = (labels * 47 % 255)[..., np.newaxis]
            cv2.rectangle(img_2, left_top, left_top + (roi_width, roi_height), color, 1)
            cv2.putText(img_2, f'{k}', left_top + np.rint([roi_width / 10, roi_height / 1.3]).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            cv2.rectangle(mask_edge, left_top, left_top + (roi_width, roi_height), color, 1)
            cv2.putText(mask_edge, f'{k}', left_top + np.rint([roi_width / 10, roi_height / 1.3]).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        util.show(img_2)
        util.show(mask_edge)
        # util.show(mask_color)

    stage = Stage(stage.reshape((h, w)))
    return stage, is_fiery


# # TODO
# def preview(img: np.ndarray, stage: Stage, rois: np.ndarray, roi_width, roi_height, debug=False) -> (Stage, Pattern):
#     h, w, _ = rois.shape
#     rois = rois.reshape((h * w, 2))
#
#     def __square_ratios(top_left: np.ndarray, hsv_ranges: Iterable) -> float:
#         roi = img[top_left[0]:top_left[0] + roi_height, top_left[1]:top_left[1] + roi_width]
#         hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#         masks = np.array([cv2.inRange(hsv, hsv_lower_bound, hsv_upper_bound) for hsv_lower_bound, hsv_upper_bound in hsv_ranges])
#         return np.sum(masks == 255, axis=(1, 2)) / (roi_width * roi_height)
#
#     ratios = np.array([__square_ratios(top_left, LIGHTER_HSV_RANGES) for top_left in rois])
#     is_wall = ratios.max(axis=1, initial=0) < SQUARE_PIXEL_RATIO
#     squares = ratios.argmax(axis=1)
#     squares[is_wall] = -1
#
#     img_2 = img.copy()
#
#     def __callback(event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             idx = -1
#             for k, top_left in enumerate(rois):
#                 if top_left[1] <= x <= top_left[1] + roi_width and top_left[0] <= y <= top_left[0] + roi_height:
#                     idx = k
#                     break
#             if idx == -1:
#                 return
#             logger.debug(f'detection.debug.roi: No.={idx}')
#             top_left = rois[idx]
#             roi = img_2[top_left[0]:top_left[0] + roi_height, top_left[1]:top_left[1] + roi_width]
#             hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#             logger.debug(f'detection.debug.roi: mean.100={np.mean(roi, axis=(0, 1))}, mean.50={np.mean(roi[:roi_height // 2], axis=(0, 1))}, , mean.25={np.mean(roi[:roi_height // 4], axis=(0, 1))}')
#
#     if debug:
#         opencv_rois = np.array([util.numpy_to_opencv(idx) for idx in rois])
#         mask = np.zeros_like(img)
#         gray = np.zeros_like(img)
#         # calculate
#         for k, top_left in enumerate(rois):
#             roi = img[top_left[0]:top_left[0] + roi_height, top_left[1]:top_left[1] + roi_width]
#             hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#             _gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#             # _gray = cv2.equalizeHist(_gray)
#             # hsv[:,:,0] = cv2.equalizeHist(hsv[:,:,0])
#             # hsv[:,:,1] = cv2.equalizeHist(hsv[:,:,1])
#             # hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
#             # hsv[:,:,0] = cv2.equalizeHist(roi[:,:,0])
#             # hsv[:,:,1] = cv2.equalizeHist(roi[:,:,1])
#             # hsv[:,:,2] = cv2.equalizeHist(roi[:,:,2])
#             edge = cv2.Canny(cv2.equalizeHist(hsv[:, :, 0]), 100, 400)
#             # edge = cv2.Canny(roi, 0, 50)
#             mask[top_left[0]:top_left[0] + roi_height, top_left[1]:top_left[1] + roi_width] = edge[..., np.newaxis]
#             gray[top_left[0]:top_left[0] + roi_height, top_left[1]:top_left[1] + roi_width] = _gray[..., np.newaxis]
#         # draw
#         for k, left_top in enumerate(opencv_rois):
#             color = DEBUG_COLOR[squares[k]]
#             cv2.rectangle(img, left_top, left_top + (roi_width, roi_height), color, 1)
#             cv2.putText(img, f'{k}', left_top + np.rint([roi_width / 10, roi_height / 1.3]).astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
#
#         util.show(img_2, __callback)
#         # util.show(img, __callback)
#         # util.show(gray, __callback)
#         util.show(mask, __callback)
#
#     return None, None
