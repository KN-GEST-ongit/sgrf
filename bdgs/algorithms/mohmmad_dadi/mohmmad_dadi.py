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
from definitions import ROOT_DIR


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
        with open(os.path.join(ROOT_DIR, "trained_models", 'mohmmad_dadi_svm.pkl'), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(ROOT_DIR, "trained_models", 'mohmmad_dadi_pca.pkl'), 'rb') as f:
            pca = pickle.load(f)

        processed_image = self.process_image(payload=payload, processing_method=processing_method).flatten()
        processed_image_pca = pca.transform([processed_image])

        predictions = model.predict(processed_image_pca)
        return GESTURE(predictions[0] + 1), 100

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

        pca = PCA(n_components=50)  # PCA can be replaced by LDA
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_pca, y_train)
        knn_accuracy = knn.score(X_test_pca, y_test)
        print(f"KNN Accuracy: {knn_accuracy * 100:.2f}%")

        svm = SVC(kernel='linear')
        svm.fit(X_train_pca, y_train)
        svm_accuracy = svm.score(X_test_pca, y_test)
        print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")

        model_path = os.path.join(target_model_path, 'mohmmad_dadi_pca.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(pca, f)
        model_path = os.path.join(target_model_path, 'mohmmad_dadi_knn.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(knn, f)
        model_path = os.path.join(target_model_path, 'mohmmad_dadi_svm.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(svm, f)

        return knn_accuracy, 0
