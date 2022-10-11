from abc import ABC, abstractmethod

import numpy as np


class ScreenCapturer(ABC):
    @abstractmethod
    def capture(self) -> np.ndarray:
        raise NotImplementedError
