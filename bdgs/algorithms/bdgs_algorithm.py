from abc import abstractmethod, ABC
from typing import Any

from cv2 import Mat
from numpy import dtype, ndarray

from bdgs.gesture import GESTURE


class BaseAlgorithm(ABC):
    @abstractmethod
    def classify(self, image: Mat | ndarray[Any, dtype]) -> GESTURE:
        """Classify gesture based on static image"""
        pass
