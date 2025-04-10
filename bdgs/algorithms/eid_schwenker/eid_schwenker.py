import os

import cv2
import keras
import numpy as np

from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.data.gesture import GESTURE
from bdgs.data.processing_method import PROCESSING_METHOD
from bdgs.models.image_payload import ImagePayload
from scripts.common.vars import TRAINED_MODELS_PATH


def segment_skin(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 0, 0], dtype=np.uint8)
    upper_skin = np.array([38, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    return cv2.bitwise_and(image, image, mask=mask)


class EidSchwenker(BaseAlgorithm):
    def process_image(self, payload: ImagePayload,
                      processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> np.ndarray:
        if processing_method == PROCESSING_METHOD.DEFAULT or processing_method == PROCESSING_METHOD.EID_SCHWENKER:
            image = payload.image

            processed = segment_skin(image)
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            processed = cv2.resize(processed, (30, 30))
            return processed
        else:
            raise Exception("Invalid processing method")

    def classify(self, payload: ImagePayload,
                 processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> (GESTURE, int):
        predicted_class = 1
        certainty = 0
        model = keras.models.load_model(os.path.join(TRAINED_MODELS_PATH, 'eid_schwenker.keras'))
        processed_image = self.process_image(payload=payload, processing_method=processing_method)
        processed_image = np.expand_dims(processed_image, axis=0)  #

        predictions = model.predict(processed_image)

        for i, prediction in enumerate(predictions):
            predicted_class = np.argmax(prediction) + 1
            certainty = int(np.max(prediction) * 100)

        return GESTURE(predicted_class), certainty
