import os
import pickle

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.data.gesture import GESTURE
from bdgs.data.processing_method import PROCESSING_METHOD
from bdgs.algorithms.gupta_jaafar.gupta_jaafar_payload import GuptaJaafarPayload
from bdgs.algorithms.gupta_jaafar.gupta_jaafar_learning_data import GuptaJaafarLearningData
from bdgs.models.learning_data import LearningData
from skimage.filters import gabor
from bdgs.common.crop_image import crop_image


class GuptaJaafar(BaseAlgorithm):
    GABOR_SCALES = [1, 2, 3]
    GABOR_ORIENTATIONS = [0, np.deg2rad(36), np.deg2rad(72), np.deg2rad(108), np.deg2rad(144)]

    def process_image(self, payload: GuptaJaafarPayload,
                      processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> np.ndarray:

        # Experimental
        if processing_method != PROCESSING_METHOD.DEFAULT:
            from bdgs.data.algorithm_functions import ALGORITHM_FUNCTIONS
            return ALGORITHM_FUNCTIONS[processing_method].process_image(payload)

        cropped_image = crop_image(payload.image, payload.coords)
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        features = []
        preview_accumulator = np.zeros_like(resized, dtype=np.float32)
        for sigma in self.GABOR_SCALES:
            for theta in self.GABOR_ORIENTATIONS:
                real, _ = gabor(resized, frequency=0.5, theta=theta, sigma_x=sigma, sigma_y=sigma)
                preview_accumulator += real.astype(np.float32)
                features.append(real.flatten())

        feature_vector = np.concatenate(features)
        preview_image = preview_accumulator / len(features)
        preview_image = cv2.normalize(preview_image, None, 0, 255, cv2.NORM_MINMAX)
        preview_image = preview_image.astype(np.uint8)
        return preview_image

    def classify(self, payload: GuptaJaafarPayload,
                 processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> (GESTURE, int):
        return 1, 100

    def learn(self, learning_data: list[GuptaJaafarLearningData], target_model_path: str) -> (float, float):
        processed_images = []
        etiquettes = []
        for data in learning_data:
            hand_image = cv2.imread(data.image_path)
            processed_image = (self.process_image(
                payload=GuptaJaafarPayload(image=hand_image, coords=data.coords)
            )).flatten()
            processed_images.append(processed_image)
            etiquettes.append(data.label.value - 1)

        X = np.array(processed_images)
        y = np.array(etiquettes)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return 0, 0
