import cv2
import numpy as np


def show(img: np.ndarray):
    cv2.imshow("debug", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def numpy_to_opencv(idx: np.ndarray) -> np.ndarray:
    return idx[::-1]


def opencv_to_numpy(idx: np.ndarray) -> np.ndarray:
    return idx[::-1]


def grid_roi_top_lefts(top_left, width, height, width_step, height_step, width_offset, height_offset):
    return np.array([(top_left[0] + int(height_step * h) + int(height_offset * w), top_left[1] + int(width_step * w) + int(width_offset * h)) for h in range(height) for w in range(width)]).reshape((height, width, 2))


def detect_cursor(img, top_lefts, width, height, hsv_lower_bound, hsv_upper_bound, threshold, debug=True):
    def __cursor_ratios(top_left: np.ndarray) -> float:
        roi = img[top_left[0]:top_left[0] + height, top_left[1]:top_left[1] + width]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower_bound, hsv_upper_bound)
        return np.sum(mask == 255) / (width * height)

    ratios = np.array([__cursor_ratios(top_left) for top_left in top_lefts])
    pos = np.argmax(ratios)
    if ratios[pos] < threshold:
        pos = -1
    if debug:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower_bound, hsv_upper_bound)
        mask = cv2.merge((mask, mask, mask))
        top_lefts = [numpy_to_opencv(idx) for idx in top_lefts]
        for i, roi in enumerate(top_lefts):
            cv2.rectangle(img, roi, roi + (width, height), (0, 255, 0), 1)
            cv2.rectangle(mask, roi, roi + (width, height), (0, 255, 0), 1)
            font_size = np.min((width, height)) / 40
            cv2.putText(mask, f'{ratios[i]:.3}', roi + (0, -5), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 1)
            cv2.putText(mask, f'{i}', roi + np.array((width / 10, height / 1.3), dtype=int), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 1)
        show(img)
        show(mask)
    return pos
