import os

import cv2
import keras
import numpy as np
from sklearn.model_selection import train_test_split

from bdgs.algorithms.murthy_jadon.murthy_jadon import MurthyJadon
from bdgs.algorithms.murthy_jadon.murthy_jadon_payload import MurthyJadonPayload
from scripts.common.get_learning_files import get_learning_files

folder_path = os.path.abspath("../../../bdgs_photos")
algorithm = MurthyJadon()


def process_image():
    images = get_learning_files()

    for image_file in images:
        image_path = os.path.join(folder_path, image_file[0])
        bg_image_path = os.path.join(folder_path, image_file[2])

        hand_image = cv2.imread(image_path)
        background_image = cv2.imread(bg_image_path)
        processed_image = algorithm.process_image(
            payload=MurthyJadonPayload(image=hand_image, bg_image=background_image))

        cv2.imshow("Processed Image", processed_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def learn():
    train_images_amount = 500
    epochs = 50

    images = get_learning_files(limit=train_images_amount, shuffle=True)
    processed_images = []
    etiquettes = []
    for image_file in images:
        image_path = os.path.join(folder_path, image_file[0])
        bg_image_path = os.path.join(folder_path, image_file[2])

        hand_image = cv2.imread(image_path)
        background_image = cv2.imread(bg_image_path)
        processed_image = (algorithm.process_image(
            payload=MurthyJadonPayload(image=hand_image, bg_image=background_image)
        ).flatten()) / 255

        processed_images.append(processed_image)
        etiquettes.append(int(image_file[1].split(" ")[0]) - 1)

    processed_images = np.array(processed_images)
    etiquettes = np.array([int(etiquette) for etiquette in etiquettes])
    X_train, X_val, y_train, y_val = train_test_split(processed_images, etiquettes, test_size=0.2, random_state=42)

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
    keras.models.save_model(model, '../../trained_models/murthy_jadon.keras')


def clasify():
    images = get_learning_files(limit=100, shuffle=True, offset=20)

    for image_file in images:
        image_path = os.path.join(folder_path, image_file[0])
        image = cv2.imread(image_path)
        background_image = cv2.imread(image_file[2])

        result = algorithm.classify(MurthyJadonPayload(image=image, bg_image=background_image))

        cv2.imshow(f"Gesture: {result}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # process_image()
    learn()
    clasify()
