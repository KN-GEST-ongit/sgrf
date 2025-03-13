import cv2
import numpy as np

from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.gesture import GESTURE


class MurthyJadon(BaseAlgorithm):
    def process_image(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            resized_empty = cv2.resize(binary_image, (30, 30))
            return cv2.bitwise_not(resized_empty)

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        hand_crop = binary_image[y:y + h, x:x + w]
        hand_resized = cv2.resize(hand_crop, (30, 30), interpolation=cv2.INTER_AREA)

        hand_inverted = cv2.bitwise_not(hand_resized)

        return hand_inverted

    def classify(self, image) -> GESTURE:
        return GESTURE.OK
