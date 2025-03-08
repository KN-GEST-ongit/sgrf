from abc import abstractmethod, ABC

from bdgs.gesture import GESTURE


class BaseAlgorithm(ABC):
    @abstractmethod
    def classify(self, image) -> GESTURE:
        """Classify gesture based on static image"""
        pass
