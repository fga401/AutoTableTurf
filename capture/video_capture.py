import cv2
import numpy as np

from capture.interface import Capture
from logger import logger


class VideoCapture(Capture):
    def __init__(self, device_idx):
        """
        Capture from the connected Camera.

        :param device_idx: Camera index.
        """
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

        def __print_debug_info(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                bgr = frame[y:y + 1, x:x + 1]
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                logger.debug(f'detection.debug: x={x}, y={y}, BGR={bgr.squeeze()}, HSV={hsv.squeeze()}')

        cv2.setMouseCallback(name, __print_debug_info)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def save(self, name):
        img = self.capture()
        cv2.imwrite(name + '.jpg', img)

    def capture(self) -> np.ndarray:
        ret, frame = self.__cam.read()
        if not ret:
            raise Exception('failed to grab frame')
        return frame

    def close(self):
        self.__cam.release()
