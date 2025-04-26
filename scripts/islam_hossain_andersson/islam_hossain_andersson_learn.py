import os

import cv2
import keras
import numpy as np
from keras import Sequential
from keras.src import layers
from keras.src.losses import CategoricalCrossentropy
from keras.src.optimizers import SGD

from sklearn.model_selection import train_test_split

from bdgs.algorithms.islam_hossain_andersson.islam_hossain_andersson import IslamHossainAndersson
from bdgs.data.gesture import GESTURE
from bdgs.algorithms.islam_hossain_andersson.islam_hossain_andersson_payload import IslamHossainAnderssonPayload
from scripts.common.crop_image import crop_image
from scripts.common.get_learning_files import get_learning_files
from scripts.common.vars import TRAINING_IMAGES_PATH, TRAINED_MODELS_PATH


def get_training_data():
    image_files = get_learning_files()
    processed_images = []
    labels = []
    i = 1

    alg = IslamHossainAndersson()

    print("Processing images...")
    for image_file in image_files:
        image_path = os.path.join(TRAINING_IMAGES_PATH, image_file[0])
        image = cv2.imread(str(image_path))
        label = int(image_file[1].split(" ")[0])
        cropped_image = crop_image(image, image_file[1])
        bg_image = cv2.imread(os.path.join(TRAINING_IMAGES_PATH, image_file[2]))
        cropped_bg = crop_image(bg_image, image_file[1])
        payload = IslamHossainAnderssonPayload(image=cropped_image, bg_image=cropped_bg)
        image = alg.process_image(payload)
        image = np.squeeze(image)

        labels.append(label - 1)
        processed_images.append(image)
        print(f'\rProcessed: {i}/{len(image_files)}', end='', flush=True)
        i+=1

    processed_images = np.array(processed_images)
    labels = np.array(labels)
    print("\nProcessing images finished.")

    return processed_images, labels

def train():
    images, labels = get_training_data()
    num_classes = len(GESTURE)

    model = create_model(num_classes, enable_augmentation=False)

    x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2,
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
        filepath=os.path.join(TRAINED_MODELS_PATH, "islam_hossain_andersson.keras")
    )

def create_model(num_classes, enable_augmentation = True):
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

if __name__ == "__main__":
    train()
