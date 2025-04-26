import os

import cv2
import keras
import numpy as np

from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.algorithms.pinto_borges.pinto_borges_payload import PintoBorgesPayload
from bdgs.data.gesture import GESTURE
from bdgs.data.processing_method import PROCESSING_METHOD
from scripts.common.crop_image import crop_image
from scripts.common.vars import TRAINED_MODELS_PATH


def skin_segmentation(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    return skin_mask


def morphological_processing(mask: np.ndarray) -> np.ndarray:
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    eroded = cv2.erode(mask, horizontal_kernel, iterations=1)

    square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, square_kernel)

    return closed


def polygonal_approximation(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approx_mask = np.zeros_like(mask)

    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(approx_mask, [approx], -1, (255,), thickness=cv2.FILLED)

    return approx_mask


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = cv2.bitwise_and(gray, gray, mask=mask)
    return result


class PintoBorges(BaseAlgorithm):
    def process_image(self, payload: PintoBorgesPayload,
                      processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> np.ndarray:
        cropped_image = crop_image(payload.image, payload.coords)

        cropped_image = cv2.resize(cropped_image, (400, 400))

        skin_mask = skin_segmentation(cropped_image)
        skin_mask = morphological_processing(skin_mask)
        skin_mask = polygonal_approximation(skin_mask)
        masked_image = apply_mask(cropped_image, skin_mask)

        return masked_image

    def classify(self, payload: PintoBorgesPayload,
                 processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> (GESTURE, int):
        predicted_class = 1
        certainty = 0
        model = keras.models.load_model(os.path.join(TRAINED_MODELS_PATH, 'pinto_borges.keras'))
        processed_image = self.process_image(payload=payload, processing_method=processing_method)
        processed_image = np.expand_dims(processed_image, axis=0)  #

        predictions = model.predict(processed_image)

        for i, prediction in enumerate(predictions):
            predicted_class = np.argmax(prediction) + 1
            certainty = int(np.max(prediction) * 100)

        return GESTURE(predicted_class), certainty
