import cv2
import numpy as np

from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.gesture import GESTURE
from bdgs.models.image_payload import ImagePayload


class Maung(BaseAlgorithm):
    def process_image(self, payload: ImagePayload) -> np.ndarray:
        gray = cv2.cvtColor(payload.image, cv2.COLOR_BGR2GRAY)

        blurred = cv2.medianBlur(gray, 15)

        resized = cv2.resize(blurred, (140, 150))

        roberts_x = np.array([[1, 0], [0, -1]])

        roberts_y = np.array([[0, 1], [-1, 0]])

        dx = cv2.filter2D(resized, cv2.CV_64F, roberts_x)
        dy = cv2.filter2D(resized, cv2.CV_64F, roberts_y)

        gradient_orientation = np.arctan2(dy, dx)

        gradient_orientation_degrees = np.degrees(gradient_orientation) % 90

        hist, _ = np.histogram(gradient_orientation_degrees, bins=3, range=(0, 90))

        return gradient_orientation_degrees

    def classify(self, image) -> GESTURE:
        return GESTURE.OK
