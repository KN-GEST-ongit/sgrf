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
from bdgs.models.image_payload import ImagePayload
from bdgs.models.learning_data import LearningData
from skimage.filters import gabor


class GuptaJaafar(BaseAlgorithm):
    def process_image(self, payload: ImagePayload,
                      processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> np.ndarray:
        # Experimental
        if processing_method != PROCESSING_METHOD.DEFAULT:
            from bdgs.data.algorithm_functions import ALGORITHM_FUNCTIONS
            return ALGORITHM_FUNCTIONS[processing_method].process_image(payload)

        gray = cv2.cvtColor(payload.image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        gabor_scales = [1, 2, 3]
        gabor_orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
        features = []
        for sigma in gabor_scales:
            for theta in gabor_orientations:
                real, _ = gabor(resized, frequency=0.2, theta=theta, sigma_x=sigma, sigma_y=sigma)
                features.append(real.flatten())
        return np.concatenate(features)

    def classify(self, payload: ImagePayload,
                 processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> (GESTURE, int):
        return 1, 100

    def learn(self, learning_data: list[LearningData], target_model_path: str) -> (float, float):
        processed_images = []
        etiquettes = []
        for data in learning_data:
            hand_image = cv2.imread(data.image_path)
            processed_image = (self.process_image(
                payload=ImagePayload(image=hand_image)
            )).flatten()
            processed_images.append(processed_image)
            etiquettes.append(data.label.value - 1)

        X = np.array(processed_images)
        y = np.array(etiquettes)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return 0, 0
