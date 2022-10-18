import cv2
import numpy as np


def show(img: np.ndarray):
    cv2.imshow("debug", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
