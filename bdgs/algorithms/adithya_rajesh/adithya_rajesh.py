import os

import cv2
import keras
import numpy as np

from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.data.gesture import GESTURE
from bdgs.data.processing_method import PROCESSING_METHOD
from bdgs.algorithms.adithya_rajesh.adithya_rajesh_payload import AdithyaRajeshPayload
from scripts.common.crop_image import crop_image
from scripts.common.vars import TRAINED_MODELS_PATH


class AdithyaRajesh(BaseAlgorithm):
    def process_image(self, payload: AdithyaRajeshPayload,
                      processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> np.ndarray:
        image = payload.image
        coords = payload.coords
        if coords is not None:
            image = crop_image(image=image, coords=coords)
        image = cv2.resize(image, (100, 100))

        return image

    def classify(self, payload: AdithyaRajeshPayload, processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> (
            GESTURE, int):
        model = keras.models.load_model(os.path.join(TRAINED_MODELS_PATH, "adithya_rajesh.keras"))
        processed_image = self.process_image(payload=payload)
        expanded_dims = np.expand_dims(processed_image, axis=0)
        predictions = model.predict(expanded_dims)

        predicted_class = 1
        certainty = 0
        for prediction in predictions:
            predicted_class = np.argmax(prediction) + 1
            certainty = int(np.max(prediction) * 100)

        return GESTURE(predicted_class), certainty
