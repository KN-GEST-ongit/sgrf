import cv2
import numpy as np

from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.gesture import GESTURE
from bdgs.models.image_payload import ImagePayload


class Maung(BaseAlgorithm):
    def process_image(self, payload: ImagePayload) -> np.ndarray:
        gray = cv2.cvtColor(payload.image, cv2.COLOR_BGR2GRAY)
        # todo - ujednolicenie tla wg artykulu
        return gray

    def classify(self, image) -> GESTURE:
        return GESTURE.OK
