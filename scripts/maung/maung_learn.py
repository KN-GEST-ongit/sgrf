import os
import pickle

import cv2
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

from bdgs.classifier import process_image
from bdgs.data.algorithm import ALGORITHM
from bdgs.models.image_payload import ImagePayload
from scripts.common.crop_image import crop_image, parse_file_coords
from scripts.common.get_learning_files import get_learning_files
from scripts.common.vars import TRAINING_IMAGES_PATH, TRAINED_MODELS_PATH


def learn():
    processed_images = []
    etiquettes = []
    images = get_learning_files(limit=1000, shuffle=True)
    for image, hand_recognition_data, _ in images:
        image_path = str(os.path.join(TRAINING_IMAGES_PATH, image))
        hand_image = crop_image(cv2.imread(image_path), parse_file_coords(hand_recognition_data))
        processed_image = (process_image(
            algorithm=ALGORITHM.MAUNG,
            payload=ImagePayload(image=hand_image)
        )).flatten()
        processed_images.append(processed_image)
        etiquettes.append(int(hand_recognition_data.split(" ")[0]) - 1)

    X = np.array(processed_images)
    y = np.array(etiquettes)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    perceptron = Perceptron(max_iter=1000, tol=1e-3)
    perceptron.fit(X_train, y_train)
    accuracy = perceptron.score(X_val, y_val)
    print(f"Accuracy on validation set: {accuracy * 100:.2f}%")
    model_path = os.path.join(TRAINED_MODELS_PATH, 'maung.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(perceptron, f)


if __name__ == "__main__":
    learn()
