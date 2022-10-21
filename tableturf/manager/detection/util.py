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
