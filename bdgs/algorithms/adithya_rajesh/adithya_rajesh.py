import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.data.gesture import GESTURE
from bdgs.data.processing_method import PROCESSING_METHOD
from bdgs.models.image_payload import ImagePayload
from scripts.common.vars import TRAINED_MODELS_PATH


class AdithyaRajesh(BaseAlgorithm):
    def process_image(self, payload: ImagePayload,
                      processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> np.ndarray:
        image = payload.image
        image = cv2.resize(image, (100, 100))
        image = image.astype(np.float32)
        # expand to get shape (1, 100, 100, 3)
        image = np.expand_dims(image, axis=0)


        return image

    def classify(self, payload: ImagePayload, processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> (GESTURE, int):
        model = load_model(os.path.join(TRAINED_MODELS_PATH, "adithya_rajesh.keras"))
        processed_image = self.process_image(payload=payload)
        predictions = model.predict(processed_image)


        for prediction in predictions:
            predicted_class = np.argmax(prediction) + 1
            certainty = int(np.max(prediction) * 100)


        return GESTURE(predicted_class), certainty
