from abc import ABC, abstractmethod

import numpy as np


class Capture(ABC):
    @abstractmethod
    def capture(self) -> np.ndarray:
        raise NotImplementedError
