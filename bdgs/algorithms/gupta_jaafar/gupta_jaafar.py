import os
import pickle

import cv2
import numpy as np
from skimage.filters import gabor
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.algorithms.gupta_jaafar.gupta_jaafar_learning_data import GuptaJaafarLearningData
from bdgs.algorithms.gupta_jaafar.gupta_jaafar_payload import GuptaJaafarPayload
from bdgs.common.crop_image import crop_image
from bdgs.data.gesture import GESTURE
from bdgs.data.processing_method import PROCESSING_METHOD
from definitions import ROOT_DIR


class GuptaJaafar(BaseAlgorithm):
    GABOR_SCALES = [1, 2, 3]
    GABOR_ORIENTATIONS = [0, np.deg2rad(36), np.deg2rad(72), np.deg2rad(108), np.deg2rad(144)]

    def __init__(self):
        self.feature_vector = None

    def process_image(self, payload: GuptaJaafarPayload,
                      processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> np.ndarray:

        # Experimental
        if processing_method != PROCESSING_METHOD.DEFAULT:
            from bdgs.data.algorithm_functions import ALGORITHM_FUNCTIONS
            return ALGORITHM_FUNCTIONS[processing_method].process_image(payload)

        cropped_image = crop_image(payload.image, payload.coords)
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (16, 16))
        features = []
        preview_accumulator = np.zeros_like(resized, dtype=np.float32)
        for sigma in self.GABOR_SCALES:
            for theta in self.GABOR_ORIENTATIONS:
                real, _ = gabor(resized, frequency=0.5, theta=theta, sigma_x=sigma, sigma_y=sigma)
                preview_accumulator += real.astype(np.float32)
                features.append(real.flatten())

        self.feature_vector = np.concatenate(features)
        preview_image = preview_accumulator / len(features)
        preview_image = cv2.normalize(preview_image, None, 0, 255, cv2.NORM_MINMAX)
        preview_image = preview_image.astype(np.uint8)
        return preview_image

    def classify(self, payload: GuptaJaafarPayload, custom_model_path=None,
                 processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> (GESTURE, int):

        model_path = custom_model_path if custom_model_path is not None else os.path.join(ROOT_DIR, "trained_models",
                                                                                          'gupta_jaafar_svm.pkl')

        with open(os.path.join(ROOT_DIR, "trained_models", 'gupta_jaafar_pca.pkl'), 'rb') as f:
            pca = pickle.load(f)
        with open(os.path.join(ROOT_DIR, "trained_models", 'gupta_jaafar_lda.pkl'), 'rb') as f:
            lda = pickle.load(f)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        self.process_image(payload=payload, processing_method=processing_method)
        pca_data = pca.transform([self.feature_vector])
        lda_data = lda.transform(pca_data)
        predictions = model.predict(lda_data)
        return GESTURE(predictions[0] + 1), 100

    def learn(self, learning_data: list[GuptaJaafarLearningData], target_model_path: str) -> (float, float):
        processed_features = []
        etiquettes = []
        for data in learning_data:
            hand_image = cv2.imread(data.image_path)
            self.process_image(payload=GuptaJaafarPayload(image=hand_image, coords=data.coords))
            processed_features.append(self.feature_vector)
            etiquettes.append(data.label.value - 1)
        X = np.array(processed_features)
        y = np.array(etiquettes)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pca = PCA(n_components=50)
        pca_data_train = pca.fit_transform(X_train)
        pca_data_test = pca.transform(X_test)
        lda = LDA(n_components=5)
        lda_data_train = lda.fit_transform(pca_data_train, y_train)
        lda_data_test = lda.transform(pca_data_test)
        svm = SVC(kernel='rbf', decision_function_shape='ovo')
        svm.fit(lda_data_train, y_train)
        train_accuracy = svm.score(lda_data_train, y_train)
        test_accuracy = svm.score(lda_data_test, y_test)
        model_path = os.path.join(target_model_path, 'gupta_jaafar_pca.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(pca, f)
        model_path = os.path.join(target_model_path, 'gupta_jaafar_lda.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(lda, f)
        model_path = os.path.join(target_model_path, 'gupta_jaafar_svm.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(svm, f)

        return train_accuracy, test_accuracy
