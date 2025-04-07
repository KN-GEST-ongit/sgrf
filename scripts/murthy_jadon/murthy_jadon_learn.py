import os

import cv2
import keras
import numpy as np
from sklearn.model_selection import train_test_split

from bdgs.algorithms.murthy_jadon.murthy_jadon_payload import MurthyJadonPayload
from bdgs.classifier import process_image
from bdgs.data.algorithm import ALGORITHM
from scripts.common.get_learning_files import get_learning_files
from scripts.common.vars import TRAINING_IMAGES_PATH, TRAINED_MODELS_PATH


def learn():
    train_images_amount = 100
    epochs = 40

    images = get_learning_files(limit=train_images_amount, shuffle=True)
    processed_images = []
    etiquettes = []
    for image_file in images:
        image_path = str(os.path.join(TRAINING_IMAGES_PATH, image_file[0]))
        bg_image_path = str(os.path.join(TRAINING_IMAGES_PATH, image_file[2]))

        hand_image = cv2.imread(image_path)
        background_image = cv2.imread(bg_image_path)
        processed_image = (process_image(
            algorithm=ALGORITHM.MURTHY_JADON,
            payload=MurthyJadonPayload(image=hand_image, bg_image=background_image)
        ).flatten()) / 255

        processed_images.append(processed_image)
        etiquettes.append(int(image_file[1].split(" ")[0]) - 1)

    X_train, X_val, y_train, y_val = train_test_split(np.array(processed_images), np.array(etiquettes), test_size=0.2,
                                                      random_state=42)

    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(900,)),
        keras.layers.Dense(14, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))
    keras.models.save_model(model, os.path.join(TRAINED_MODELS_PATH, 'murthy_jadon.keras'))


if __name__ == '__main__':
    learn()
