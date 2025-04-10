import os

import cv2
import keras
import numpy as np
from keras.api import layers, models
from sklearn.model_selection import train_test_split

from bdgs.classifier import process_image
from bdgs.data.algorithm import ALGORITHM
from bdgs.models.image_payload import ImagePayload
from scripts.common.get_learning_files import get_learning_files
from scripts.common.vars import TRAINING_IMAGES_PATH, TRAINED_MODELS_PATH


def learn():
    train_images_amount = 1000
    epochs = 100

    images = get_learning_files(limit=train_images_amount, shuffle=True)
    processed_images = []
    etiquettes = []

    for image_file in images:
        image_path = os.path.join(TRAINING_IMAGES_PATH, image_file[0])

        hand_image = cv2.imread(str(image_path))
        processed_image = process_image(
            algorithm=ALGORITHM.EID_SCHWENKER,
            payload=ImagePayload(image=hand_image)
        )

        processed_images.append(processed_image)
        etiquettes.append(int(image_file[1].split(" ")[0]) - 1)

    X = np.array(processed_images).reshape(-1, 30, 30, 1)
    y = np.array(etiquettes)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = models.Sequential([
        layers.Conv2D(15, (6, 6), activation='relu', input_shape=(30, 30, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(30, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))
    keras.models.save_model(model, os.path.join(TRAINED_MODELS_PATH, 'eid_schwenker.keras'))


if __name__ == '__main__':
    learn()
