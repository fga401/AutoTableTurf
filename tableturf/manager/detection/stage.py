import cv2
import numpy as np

from tableturf.manager.detection import util
from tableturf.manager.detection.ui import special_on
from tableturf.model import Stage, Pattern

BOUNDING_BOX_TOP_LEFT = np.array([0, 750])
BOUNDING_BOX_WIDTH = 800
BOUNDING_BOX_HEIGHT = 1080

CC_WIDTH_UPPER_BOUND = 45
CC_WIDTH_LOWER_BOUND = 30
CC_HEIGHT_UPPER_BOUND = 45
CC_HEIGHT_LOWER_BOUND = 30
CC_AREA_UPPER_BOUND = CC_WIDTH_UPPER_BOUND * CC_HEIGHT_UPPER_BOUND
CC_AREA_LOWER_BOUND = CC_WIDTH_LOWER_BOUND * CC_HEIGHT_LOWER_BOUND

EMPTY_COLOR_HSV_UPPER_BOUND = (180, 255, 46)
EMPTY_COLOR_HSV_LOWER_BOUND = (0, 0, 0)
ROI_EROSION_SIZE = 10


def kmeans(data: np.ndarray, k=3, normalize=False, limit=500):
    if normalize:
        stats = (data.mean(axis=0), data.std(axis=0))
        data = (data - stats[0]) / stats[1]
    centers = data[:k]

    for i in range(limit):
        classifications = np.argmin(((data[..., np.newaxis] - centers.T[np.newaxis, ...]) ** 2).sum(axis=1), axis=1)
        new_centers = np.array([data[classifications == j].mean(axis=0) for j in range(k)])
        if np.all(new_centers == centers):
            break
        else:
            centers = new_centers
    else:
        raise RuntimeError(f'Clustering algorithm did not complete within {limit} iterations')

    if normalize:
        centers = centers * stats[1] + stats[0]
    return classifications, centers


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


def stage(img: np.ndarray, rois: np.ndarray, debug=False) -> (Stage, Pattern):
    """
    :param rois: (h, w, 2), rois[i][j] = (y, x)
    :return: stage and pattern preview
    """
    sp_on = special_on(img, debug)
    h, w, _ = rois.shape
    stage = np.zeros((h, w))
    preview = np.zeros((h, w))
    # TODO
    return Stage(stage), Pattern(preview)
