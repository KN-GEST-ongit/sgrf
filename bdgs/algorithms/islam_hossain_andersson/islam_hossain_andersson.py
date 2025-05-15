import os

import cv2
import keras
import numpy as np
from keras import Sequential
from keras.src import layers
from keras.src.losses import CategoricalCrossentropy
from keras.src.optimizers import SGD
from sklearn.model_selection import train_test_split

from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.algorithms.islam_hossain_andersson.islam_hossain_andersson_learning_data import \
    IslamHossainAnderssonLearningData
from bdgs.algorithms.islam_hossain_andersson.islam_hossain_andersson_payload import IslamHossainAnderssonPayload
from bdgs.common.crop_image import crop_image
from bdgs.data.gesture import GESTURE
from bdgs.data.processing_method import PROCESSING_METHOD
from definitions import ROOT_DIR


def create_model(num_classes, enable_augmentation=True):
    model = Sequential()

    # augmentation parameter values were not specified, so they were found with experiments.
    if enable_augmentation:
        # augmentation layer
        model.add(keras.Sequential([
            layers.Rescaling(1.0 / 255, input_shape=(50, 50, 1)),
            layers.RandomRotation(0.10),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomShear(0.1),
            layers.RandomFlip("horizontal")
        ]))
    else:
        model.add(layers.Rescaling(1.0 / 255, input_shape=(50, 50, 1)))

    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(
        optimizer=SGD(learning_rate=0.001),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model


class IslamHossainAndersson(BaseAlgorithm):
    def process_image(self, payload: IslamHossainAnderssonPayload,
                      processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> np.ndarray:
        if processing_method == PROCESSING_METHOD.DEFAULT or processing_method == PROCESSING_METHOD.ISLAM_HOSSAIN_ANDERSSON:
            image = payload.image
            background = payload.bg_image
            if payload.coords is not None:
                image = crop_image(image, payload.coords)
                background = crop_image(background, payload.coords)
            # The paper did not specify exact parameters for preprocessing methods, so
            # values used here were selected based on experiments to achieve best results

            # Zoran Zivkovic method to subtract the background. 
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=2, varThreshold=16)
            bg_subtractor.apply(background)
            fg_mask = bg_subtractor.apply(image)
            fg_color = cv2.bitwise_and(image, image, mask=fg_mask)
            # grayscale
            grayscale = cv2.cvtColor(fg_color, cv2.COLOR_BGR2GRAY)
            # morphological erosion
            kernel = np.ones((5, 5), np.uint8)
            erosion = cv2.erode(grayscale, kernel)
            # median filter
            median_filter = cv2.medianBlur(erosion, 5)
            # resize
            resized = cv2.resize(median_filter, (50, 50))

            return resized
        else:
            raise Exception("Invalid processing method")

    def classify(self, payload: IslamHossainAnderssonPayload, custom_model_dir=None,
                 processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> (
            GESTURE, int):

        model_filename = "islam_hossain_andersson.keras"
        model_path = os.path.join(custom_model_dir, model_filename) if custom_model_dir is not None else os.path.join(
            ROOT_DIR, "trained_models",
            model_filename)

        model = keras.models.load_model(model_path)
        processed_image = self.process_image(payload=payload)
        expanded_dims = np.expand_dims(processed_image, axis=0)
        predictions = model.predict(expanded_dims, verbose=0)

        predicted_class = 1
        certainty = 0
        for prediction in predictions:
            predicted_class = np.argmax(prediction) + 1
            certainty = int(np.max(prediction) * 100)

        return GESTURE(predicted_class), certainty

    def learn(self, learning_data: list[IslamHossainAnderssonLearningData], target_model_path: str) -> (float, float):
        processed_images = []
        labels = []
        for data in learning_data:
            hand_image = cv2.imread(data.image_path)
            background_image = cv2.imread(data.bg_image_path)
            processed_image = self.process_image(
                payload=IslamHossainAnderssonPayload(image=hand_image, coords=data.coords, bg_image=background_image))

            processed_images.append(processed_image)
            labels.append(data.label.value - 1)

        processed_images = np.array(processed_images)
        labels = np.array(labels)
        num_classes = len(GESTURE)

        model = create_model(num_classes, enable_augmentation=False)

        x_train, x_val, y_train, y_val = train_test_split(processed_images, labels, test_size=0.2,
                                                          random_state=42)
        y_train_one_hot = keras.utils.to_categorical(y_train, num_classes=num_classes)
        y_val_one_hot = keras.utils.to_categorical(y_val, num_classes=num_classes)

        history = model.fit(x_train, y_train_one_hot,
                            validation_data=(x_val, y_val_one_hot),
                            batch_size=32,
                            epochs=60,
                            verbose="auto")

        keras.models.save_model(
            model=model,
            filepath=os.path.join(target_model_path, "islam_hossain_andersson.keras")
        )

        test_loss, test_acc = model.evaluate(x_val, y_val_one_hot, verbose=0)
        return test_acc, test_loss
