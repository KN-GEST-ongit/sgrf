import os.path

import cv2
import keras
import numpy as np

from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.algorithms.murthy_jadon.murthy_jadon_payload import MurthyJadonPayload
from bdgs.gesture import GESTURE


def subtract_background(background, image):
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=2, varThreshold=50, detectShadows=False)
    bg_subtractor.apply(background)
    fg_mask = bg_subtractor.apply(image)

    kernel = np.ones((3, 3), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)
    _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
    fg_color = cv2.bitwise_and(image, image, mask=fg_mask)

    return fg_color


def extract_hand_region(image):
    image = image.astype(np.uint8)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, (0, 0, 0, 0)
    largest_contour = max(contours, key=cv2.contourArea)

    hand_mask = np.zeros_like(image, dtype=np.uint8)
    cv2.drawContours(hand_mask, [largest_contour], -1, [255], thickness=cv2.FILLED)

    x, y, w, h = cv2.boundingRect(largest_contour)
    hand_region = hand_mask[y:y + h, x:x + w]

    cut_ratio = 0.2
    cut_height = int(h * (1 - cut_ratio))
    hand_region = hand_region[:cut_height, :]

    result = cv2.bitwise_not(hand_region)
    return result


class MurthyJadon(BaseAlgorithm):
    def process_image(self, payload: MurthyJadonPayload) -> np.ndarray:
        image = payload.image
        background = payload.bg_image

        subtracted = subtract_background(background, image)
        gray = cv2.cvtColor(subtracted, cv2.COLOR_BGR2GRAY)
        hand_only = extract_hand_region(gray)
        resized = cv2.resize(hand_only, (30, 30))

        return resized

    def classify(self, payload: MurthyJadonPayload) -> GESTURE:
        predicted_class = 1
        model = keras.models.load_model(os.path.join('../../trained_models', 'murthy_jadon.keras'))
        processed_image = (self.process_image(payload=payload).flatten()) / 255
        processed_image = np.expand_dims(processed_image, axis=0)  #

        print(processed_image.shape)

        predictions = model.predict(processed_image)

        for i, prediction in enumerate(predictions):
            predicted_class = np.argmax(prediction) + 1

        return GESTURE(predicted_class)
