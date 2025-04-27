import os
import cv2
import numpy as np
import pickle

from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.data.processing_method import PROCESSING_METHOD
from bdgs.models.image_payload import ImagePayload
from bdgs.data.gesture import GESTURE
from scripts.common.vars import TRAINED_MODELS_PATH
from sklearn.decomposition import PCA


class MohmmadDadi(BaseAlgorithm):
    def process_image(self, payload: ImagePayload,
                      processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> np.ndarray:
        # Experimental
        if processing_method != PROCESSING_METHOD.DEFAULT:
            from bdgs.data.algorithm_functions import ALGORITHM_FUNCTIONS
            return ALGORITHM_FUNCTIONS[processing_method].process_image(payload)

        gray = cv2.cvtColor(payload.image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100, 100))  # added
        _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=2)
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        processed = cv2.subtract(binary, dilated)
        edges = cv2.Canny(processed, threshold1=100, threshold2=200)
        return edges

    def classify(self, payload: ImagePayload,
                 processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> (GESTURE, int):

        with open(os.path.join(TRAINED_MODELS_PATH, 'mohmmad_dadi_svm.pkl'), 'rb') as f:
            model = pickle.load(f)

        # with open(os.path.join(TRAINED_MODELS_PATH, 'mohmmad_dadi_knn.pkl'), 'rb') as f:
        #     model = pickle.load(f)

        with open(os.path.join(TRAINED_MODELS_PATH, 'mohmmad_dadi_pca.pkl'), 'rb') as f:
            pca = pickle.load(f)

        processed_image = self.process_image(payload=payload, processing_method=processing_method).flatten()
        processed_image_pca = pca.transform([processed_image])

        predictions = model.predict(processed_image_pca)
        return GESTURE(predictions[0] + 1), 100

