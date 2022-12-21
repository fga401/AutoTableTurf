from abc import ABC

import numpy as np


class Debugger(ABC):
    def show(self, name: str, img: np.ndarray):
        raise NotImplementedError

    def __bool__(self):
        return True
