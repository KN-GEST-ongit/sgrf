import os
import cv2
import keras
import numpy as np
from keras.api import layers, models
from sklearn.model_selection import train_test_split
from bdgs.algorithms.pinto_borges.pinto_borges_payload import PintoBorgesPayload
from bdgs.classifier import process_image
from bdgs.data.algorithm import ALGORITHM
from scripts.common.get_learning_files import get_learning_files
from scripts.common.vars import TRAINING_IMAGES_PATH, TRAINED_MODELS_PATH

def learn():
    train_images_amount = 100
    epochs = 10

    images = get_learning_files(limit=train_images_amount, shuffle=True)
    processed_images = []
    etiquettes = []

    for image_file in images:
        image_path = os.path.join(TRAINING_IMAGES_PATH, image_file[0])
        hand_image = cv2.imread(str(image_path))
        processed_image = process_image(
            algorithm=ALGORITHM.PINTO_BORGES,
            payload=PintoBorgesPayload(image=hand_image, coords=image_file[1])
        )

        processed_images.append(processed_image)
        etiquettes.append(int(image_file[1].split(" ")[0]) - 1)

    X = np.array(processed_images).reshape(-1, 400, 400, 1)
    y = np.array(etiquettes)

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.05, random_state=42)

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(400, 400, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(400, activation='relu'),
        layers.Dense(800, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), verbose=2, batch_size=8)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    keras.models.save_model(model, os.path.join(TRAINED_MODELS_PATH, 'pinto_borges.keras'))

if __name__ == '__main__':
    learn()
