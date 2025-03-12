from abc import abstractmethod, ABC
from typing import Any

from cv2 import Mat
from numpy import dtype, ndarray

from bdgs.gesture import GESTURE


class BaseAlgorithm(ABC):
    @abstractmethod
    def process_image(self, image: ndarray) -> ndarray:
        """Process image"""
        raise NotImplementedError("Method process_image not implemented")

    @abstractmethod
    def classify(self, image: Mat | ndarray[Any, dtype]) -> GESTURE:
        """Classify gesture based on static image"""
        raise NotImplementedError("Method classify not implemented")
