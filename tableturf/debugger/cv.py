import cv2
import numpy as np

from logger import logger
from tableturf.debugger.interface import Debugger


class OpenCVDebugger(Debugger):
    def show(self, name: str, img: np.ndarray):
        def __print_debug_info(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                bgr = img[y:y + 1, x:x + 1]
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                logger.debug(f'detection.debug: x={x}, y={y}, BGR={bgr.squeeze()}, HSV={hsv.squeeze()}')

        cv2.imshow(name, img)
        cv2.setMouseCallback(name, __print_debug_info)
        cv2.waitKey()
        cv2.destroyAllWindows()
