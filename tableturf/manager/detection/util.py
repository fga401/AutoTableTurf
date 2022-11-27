from typing import Optional

import cv2
import numpy as np

from tableturf.debugger.interface import Debugger


def numpy_to_opencv(idx: np.ndarray) -> np.ndarray:
    return idx[::-1]


def opencv_to_numpy(idx: np.ndarray) -> np.ndarray:
    return idx[::-1]


def grid_roi_top_lefts(top_left, width, height, width_step, height_step, width_offset, height_offset):
    return np.array([(top_left[0] + np.round(height_step * h) + np.round(height_offset * w), top_left[1] + np.round(width_step * w) + np.round(width_offset * h)) for h in range(height) for w in range(width)]).astype(int).reshape((height, width, 2))


def detect_cursor(img: np.ndarray, top_lefts, width, height, hsv_ranges, threshold, debug: Optional[Debugger] = None, debug_prefix: str = ''):
    def __cursor_ratio(top_left: np.ndarray) -> float:
        # print(classify_color(img[top_left[0]:top_left[0] + height, top_left[1]:top_left[1] + width], k=2))
        roi = img[top_left[0]:top_left[0] + height, top_left[1]:top_left[1] + width]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        masks = [cv2.inRange(hsv, lower_bound, upper_bound) for (lower_bound, upper_bound) in hsv_ranges]
        mask = np.max(masks, axis=0)
        return np.sum(mask == 255) / (width * height)

    ratios = np.array([__cursor_ratio(top_left) for top_left in top_lefts])
    pos = np.argmax(ratios)
    if ratios[pos] < threshold:
        pos = -1
    if debug:
        img2 = img.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        masks = [cv2.inRange(hsv, lower_bound, upper_bound) for (lower_bound, upper_bound) in hsv_ranges]
        mask = np.max(masks, axis=0)
        mask = cv2.merge((mask, mask, mask))
        top_lefts = [numpy_to_opencv(idx) for idx in top_lefts]
        for i, roi in enumerate(top_lefts):
            cv2.rectangle(img2, roi, roi + (width, height), (0, 255, 0), 1)
            cv2.rectangle(mask, roi, roi + (width, height), (0, 255, 0), 1)
            font_size = np.min((width, height)) / 40
            cv2.putText(mask, f'{ratios[i]:.3}', roi + (0, -5), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 1)
            cv2.putText(mask, f'{i}', roi + np.rint([width / 10, height / 1.3]).astype(int), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 1)
        debug.show(f'{debug_prefix}.image', img2)
        debug.show(f'{debug_prefix}.color_mask', mask)
    return pos


def kmeans(data: np.ndarray, k=3, normalize=False, limit=5000):
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


def classify_color(roi: np.ndarray, k=5):
    h, w, _ = roi.shape
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    classifications, centers = kmeans(hsv.reshape((-1, 3)), k, normalize=False)
    return classifications.reshape((h, w)), centers
