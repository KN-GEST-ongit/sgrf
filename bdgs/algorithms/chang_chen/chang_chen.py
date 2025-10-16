import os
import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
import mahotas

from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.data.gesture import GESTURE
from bdgs.data.processing_method import PROCESSING_METHOD
from bdgs.models.image_payload import ImagePayload
from bdgs.models.learning_data import LearningData
from definitions import ROOT_DIR


def compute_zernike_features_from_mask(mask_255: np.ndarray, degree: int = 7):
    mask = (mask_255 > 0).astype(np.uint8)

    ys, xs = np.where(mask)
    if ys.size == 0:
        approx_len = (degree + 1) * (degree + 1)
        return np.zeros((approx_len,), dtype=np.float32)

    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()
    w = maxx - minx + 1
    h = maxy - miny + 1
    side = max(w, h)

    R = int(np.ceil(side / 2.0))
    R = max(32, min(R, 256))
    canvas_size = 2 * R

    roi = mask[miny:maxy + 1, minx:maxx + 1].astype(np.uint8)

    if roi.shape[0] > canvas_size or roi.shape[1] > canvas_size:
        roi = cv2.resize(roi, (canvas_size, canvas_size), interpolation=cv2.INTER_NEAREST)
        oy, ox = 0, 0
    else:
        oy = (canvas_size - roi.shape[0]) // 2
        ox = (canvas_size - roi.shape[1]) // 2

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    canvas[oy:oy + roi.shape[0], ox:ox + roi.shape[1]] = roi

    img_float = (canvas > 0).astype(np.float32)
    try:
        zm = mahotas.features.zernike_moments(img_float, R, degree)
    except Exception:
        approx_len = (degree + 1) * (degree + 1)
        zm = np.zeros((approx_len,), dtype=np.float32)

    return np.array(zm, dtype=np.float32)



class ChangChen(BaseAlgorithm):
    def process_image(self, payload: ImagePayload,
                      processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> np.ndarray:
        image = payload.image
        if image is None:
            return np.zeros((1,), dtype=np.float32)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_skin = np.array([0, 20, 60])
        upper_skin = np.array([25, 255, 255])
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.medianBlur(mask, 5)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros((1,), dtype=np.float32)
        cnt = max(contours, key=cv2.contourArea)

        if cv2.contourArea(cnt) < 100:
            return np.zeros((1,), dtype=np.float32)

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        cx, cy, radius = int(cx), int(cy), int(np.ceil(radius))

        if radius < 5:
            radius = 5
        max_allowed_R = 256
        if radius > max_allowed_R:
            radius = max_allowed_R

        x, y, w, h = cv2.boundingRect(cnt)
        roi_mask = mask[y:y + h, x:x + w]

        rel_cx = cx - x
        rel_cy = cy - y
        palm_mask_roi = np.zeros_like(roi_mask)
        palm_rad = int(np.ceil(radius * 0.6))
        cv2.circle(palm_mask_roi, (rel_cx, rel_cy), palm_rad, 255, -1)

        palm_roi = cv2.bitwise_and(roi_mask, palm_mask_roi)
        fingers_roi = cv2.bitwise_and(roi_mask, cv2.bitwise_not(palm_mask_roi))

        zm_fingers = compute_zernike_features_from_mask(fingers_roi, degree=7)
        zm_palm = compute_zernike_features_from_mask(palm_roi, degree=7)

        Wfinger, Wpalm = 0.7, 0.3

        L = max(len(zm_fingers), len(zm_palm))
        if len(zm_fingers) < L:
            zm_fingers = np.pad(zm_fingers, (0, L - len(zm_fingers)))
        if len(zm_palm) < L:
            zm_palm = np.pad(zm_palm, (0, L - len(zm_palm)))

        combined = np.concatenate([Wfinger * zm_fingers, Wpalm * zm_palm]).astype(np.float32)
        return combined

    def classify(self, payload: ImagePayload, custom_model_dir=None,
                 processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> (GESTURE, int):
        model_filename = "chang_chen.pkl"
        model_path = os.path.join(custom_model_dir, model_filename) if custom_model_dir is not None else os.path.join(
            ROOT_DIR, "trained_models",
            model_filename)
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        features = self.process_image(payload, processing_method).reshape(1, -1)
        prediction = model.predict(features)
        distances, indices = model.kneighbors(features, n_neighbors=1)
        confidence = round(100*(1 / (1 + distances[0][0])), 0)

        return GESTURE(prediction[0] + 1), confidence

    def learn(self, learning_data: list[LearningData], target_model_path: str) -> (float, float):
        X, y = [], []
        for data in learning_data:
            image = cv2.imread(data.image_path)
            features = self.process_image(ImagePayload(image))
            if features is not None and features.size > 0:
                X.append(features)
                y.append(data.label.value - 1)

        X, y = np.array(X), np.array(y)
        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(X, y)
        accuracy = model.score(X, y)

        model_path = os.path.join(target_model_path, "chang_chen.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        return accuracy, None
