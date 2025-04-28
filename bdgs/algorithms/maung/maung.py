import os
import pickle

import cv2
import numpy as np

from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.data.gesture import GESTURE
from bdgs.data.processing_method import PROCESSING_METHOD
from bdgs.models.image_payload import ImagePayload
from scripts.common.vars import TRAINED_MODELS_PATH


class Maung(BaseAlgorithm):
    def process_image(self, payload: ImagePayload,
                      processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> np.ndarray:
        # Experimental
        if processing_method != PROCESSING_METHOD.DEFAULT:
            from bdgs.data.algorithm_functions import ALGORITHM_FUNCTIONS
            return ALGORITHM_FUNCTIONS[processing_method].process_image(payload)

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

        return np.float32(gradient_orientation_degrees)  # default without float32 conversion (only for cam_test)
        # return hist.astype(np.float32)

    def classify(self, payload: ImagePayload,
                 processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> (GESTURE, int):
        predicted_class = 1
        certainty = 0
        with open(os.path.join(TRAINED_MODELS_PATH, 'maung.pkl'), 'rb') as f:
            model = pickle.load(f)
        processed_image = (self.process_image(payload=payload, processing_method=processing_method)).flatten()
        processed_image = np.expand_dims(processed_image, axis=0)  #
        predictions = model.predict(processed_image)
        return GESTURE(predictions[0] + 1), 100
