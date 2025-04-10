from abc import abstractmethod, ABC

from numpy import ndarray

from bdgs.data.gesture import GESTURE
from bdgs.data.processing_method import PROCESSING_METHOD
from bdgs.models.image_payload import ImagePayload


class BaseAlgorithm(ABC):
    @staticmethod
    @abstractmethod
    def process_image(payload: ImagePayload,
                      processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> ndarray:
        """Process image"""
        raise NotImplementedError("Method process_image not implemented")

    @abstractmethod
    def classify(self, payload: ImagePayload,
                 processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> GESTURE:
        """Classify gesture based on static image"""
        raise NotImplementedError("Method classify not implemented")
