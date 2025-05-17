import os

import cv2
from numpy import ndarray
import keras
import numpy as np
from keras import Sequential
from keras.src.layers import Rescaling, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.src.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.src.utils import to_categorical

from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.data.gesture import GESTURE
from bdgs.data.processing_method import PROCESSING_METHOD
from bdgs.algorithms.mohanty_rambhatla.mohanty_rambhatla_payload import MohantyRambhatlaPayload
from bdgs.models.learning_data import LearningData
from bdgs.common.crop_image import crop_image

def augment(image: ndarray, repeat_num: int, target_size: tuple[int, int] = (32, 32)):
    images = []

    for i in range(1, repeat_num + 1):
        cropped = image[i:, i:]
        resized = cv2.resize(cropped, target_size)
        images.append(resized)

    return images

def create_model(learning_rate: float, use_relu: bool,
                num_classes: int, dropout_rate: float = 0.5):
    model = Sequential()
    model.add(Rescaling(1.0 / 255))

    model.add(Conv2D(filters=10, kernel_size=(5, 5)))
    model.add(Activation('relu' if use_relu else 'sigmoid'))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=20, kernel_size=(5, 5)))
    model.add(Activation('relu' if use_relu else 'sigmoid'))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='relu' if use_relu else 'sigmoid'))


    model.compile(
        optimizer=SGD(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=["accuracy"],
    )
    return model

class MohantyRambhatla(BaseAlgorithm):
    def process_image(self, payload: MohantyRambhatlaPayload,
                      processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> ndarray:
        image = payload.image
        coords = payload.coords
        if coords is not None:
            image = crop_image(image=image, coords=coords)
        image = cv2.resize(image, (32, 32))

        return image

    def classify(self, payload: MohantyRambhatlaPayload, custom_model_path = None,
                 processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> GESTURE:
       pass

    def learn(self, learning_data: list[LearningData], target_model_path: str
              )-> (float, float):
        enable_augmentation = False
        learning_rate = 0.01
        epochs = 10
        batch_size = 10
        use_relu = True
        dropout_rate = 0.5

        processed_images = []
        labels = []
        for data in learning_data:
            hand_image = cv2.imread(data.image_path)

            processed_image = self.process_image(
                payload=MohantyRambhatlaPayload(image=hand_image, coords=data.coords))

            if enable_augmentation:
                augmented_images = augment(hand_image, 5)
                for augmented_image in augmented_images:
                    processed_images.append(augmented_image)
                    labels.append(data.label.value - 1)
            else:
                processed_images.append(processed_image)
                labels.append(data.label.value - 1)

        processed_images = np.array(processed_images)
        labels = np.array(labels)

        num_classes = len(GESTURE)

        model = create_model(learning_rate=learning_rate, 
                             use_relu=use_relu, num_classes=num_classes,
                             dropout_rate=dropout_rate)

        x_train, x_val, y_train, y_val = train_test_split(processed_images, labels, test_size=0.2,
                                                          random_state=42)

        y_train = to_categorical(y_train, num_classes=num_classes)
        y_val = to_categorical(y_val, num_classes=num_classes)

        history = model.fit(x_train, y_train,
                            validation_data=(x_val, y_val),
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose="auto")

        keras.models.save_model(
            model=model,
            filepath=os.path.join(target_model_path, "mohanty_rambhatla.keras")
        )
        test_loss, test_acc = model.evaluate(x_val, y_val, verbose=0)

        return test_acc, test_loss
        
