import cv2
import numpy as np

from capture.interface import Capture


class VideoCapture(Capture):
    def __init__(self, device_idx):
        self.__cam = cv2.VideoCapture(device_idx)

    @property
    def width(self):
        return self.__cam.get(cv2.CAP_PROP_FRAME_WIDTH)

    @property
    def height(self):
        return self.__cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def show(self, name='image'):
        cv2.namedWindow(name)
        ret, frame = self.__cam.read()
        if not ret:
            raise Exception('failed to grab frame')
        cv2.imshow(name, frame)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def capture(self) -> np.ndarray:
        ret, frame = self.__cam.read()
        if not ret:
            raise Exception('failed to grab frame')
        return frame
