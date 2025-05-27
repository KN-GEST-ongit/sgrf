import os

import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

from bdgs.algorithms.zhuang_yang.zhuang_yang_learning_data import ZhuangYangLearningData
from bdgs.algorithms.zhuang_yang.zhuang_yang_payload import ZhuangYangPayload
from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.common.crop_image import crop_image
from bdgs.data.gesture import GESTURE
from bdgs.data.processing_method import PROCESSING_METHOD
from definitions import ROOT_DIR


class ZhuangYang(BaseAlgorithm):
    def process_image(self, payload: ZhuangYangPayload,
                      processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> np.ndarray:
        image = payload.image
        coords = payload.coords

        if coords is not None:
            image = crop_image(image=image, coords=coords)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert into gray
        image = cv2.resize(image, (80, 80))  # resize to 80x80
        image = image.astype(np.float32) / 255.0  # normalize values of pixels from 0-255 to 0-1

        return image

    def classify(self, payload: ZhuangYangPayload, custom_model_path=None,
                 processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> (GESTURE, int):
        model_filename = "zhuang_yang.pkl"
        model_path = os.path.join(custom_model_path, model_filename) if custom_model_path is not None else os.path.join(
            ROOT_DIR, "trained_models",
            model_filename)
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        W_pinv = model["W_pinv"]
        Y_train = model["Y_train"]
        labels_train = model["labels_train"]
        phi = model["phi"]
        sparsity_level = model["sparsity_level"]

        processed_image = self.process_image(payload)
        image_vector = image_to_vector(processed_image)

        y_0 = W_pinv @ image_vector

        pred_label, certainty = cs_classify_test_sample(y_0, Y_train, labels_train, phi, sparsity_level)
        pred_label += 1

        return GESTURE(pred_label), certainty

    def learn(self, learning_data: list[ZhuangYangLearningData], target_model_path: str) -> (float, float):
        processed_images = []
        labels = []
        for data in learning_data:
            hand_image = cv2.imread(data.image_path)
            processed_image = self.process_image(
                payload=ZhuangYangPayload(image=hand_image, coords=data.coords))

            image_vector = image_to_vector(processed_image)

            processed_images.append(image_vector)
            labels.append(data.label.value - 1)

        processed_images = np.stack(processed_images)
        processed_images = processed_images.reshape(-1, 6400)
        labels = np.array(labels)

        processed_images = processed_images.reshape(processed_images.shape[0], -1)
        x_train, x_test, labels_train, labels_test = train_test_split(processed_images, labels, test_size=0.2,
                                                                      random_state=42, stratify=labels)

        V_train = x_train.T
        V_test = x_test.T  # x training images

        r = 60
        max_iter = 100

        W = np.random.rand(6400, r)
        H = np.random.rand(r, len(x_train))

        W, H = nmf_update(V_train, W, H, max_iter)

        W_pinv = np.linalg.pinv(W)

        Y_train = W_pinv @ V_train
        Y_test = W_pinv @ V_test

        d = 50
        phi = np.random.randn(d, r)

        sparsity_level = 10
        predictions = []
        for i in range(Y_test.shape[1]):
            y_0 = Y_test[:, i].reshape(-1, 1)
            pred, certainty = cs_classify_test_sample(y_0, Y_train, labels_train, phi, sparsity_level)
            predictions.append(pred)

        model = {
            "W_pinv": W_pinv,
            "Y_train": Y_train,
            "labels_train": labels_train,
            "phi": phi,
            "sparsity_level": sparsity_level
        }
        filepath = os.path.join(target_model_path, "zhuang_yang.pkl")
        with open(filepath, "wb") as f:
            pickle.dump(model, f)

        accuracy = float(np.mean(np.array(predictions) == labels_test))
        loss = 1 - accuracy

        return accuracy, loss


def image_to_vector(image):
    return image.flatten().reshape(-1, 1)  # flatten to 6400x1 column vector


def nmf_update(v, w, h, max_iter=100, epsilon=1e-10):
    for _ in range(max_iter):
        WH = np.dot(w, h) + epsilon

        w *= (np.dot(v, h.T)) / (np.dot(WH, h.T) + epsilon)  # nominator_W / denominator_W
        h *= (np.dot(w.T, v)) / (np.dot(w.T, WH) + epsilon)  # nominator_H / denominator_H
        norms = np.linalg.norm(w, axis=0) + epsilon
        w /= norms

    return w, h


def omp(a, y0, sparsity_level=10):
    d, m, = a.shape
    residual = y0.copy()
    idx_selected = []
    theta_ls = []
    theta = np.zeros((m, 1))

    for _ in range(sparsity_level):
        correlations = a.T @ residual
        idx = np.argmax(np.abs(correlations))

        if idx not in idx_selected:
            idx_selected.append(idx)

        A_selected = a[:, idx_selected]
        theta_ls, _, _, _ = np.linalg.lstsq(A_selected, y0, rcond=None)

        residual = y0 - A_selected @ theta_ls

        if np.linalg.norm(residual) < 1e-5:
            break

    for i, idx in enumerate(idx_selected):
        theta[idx] = theta_ls[i]

    return theta


def classify_by_reconstruction_error(a, y0, theta, labels_train):
    classes = np.unique(labels_train)
    errors = []

    for c in classes:
        mask = (labels_train == c).astype(float).reshape(-1, 1)
        theta_i = theta * mask

        Y_i = a @ theta_i

        error = np.linalg.norm(y0 - Y_i)
        errors.append(error)

    best_class = classes[np.argmin(errors)]
    return best_class, errors


def cs_classify_test_sample(y_0, y_train, labels_train, phi, sparsity_level=10):
    A = phi @ y_train  # (d, m)
    Y0 = phi @ y_0  # (d, 1)

    theta = omp(A, Y0, sparsity_level)
    predicted_label, errors = classify_by_reconstruction_error(A, Y0, theta, labels_train)

    total_error = np.sum(errors)
    predicted_error = errors[np.argmin(errors)]
    certainty = 1.0 - predicted_error / total_error

    return predicted_label, certainty
