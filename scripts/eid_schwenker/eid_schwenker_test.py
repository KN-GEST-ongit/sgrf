import os

import cv2

from bdgs.classifier import process_image, classify
from bdgs.data.algorithm import ALGORITHM
from bdgs.data.processing_method import PROCESSING_METHOD
from bdgs.models.image_payload import ImagePayload
from scripts.common.camera_test import camera_test
from scripts.common.get_learning_files import get_learning_files
from scripts.common.vars import TRAINING_IMAGES_PATH


def process_image_test():
    images = get_learning_files()

    for image_file in images:
        image_path = str(os.path.join(TRAINING_IMAGES_PATH, image_file[0]))

        hand_image = cv2.imread(image_path)
        processed_image = process_image(
            algorithm=ALGORITHM.EID_SCHWENKER,
            payload=ImagePayload(image=hand_image),
            processing_method=PROCESSING_METHOD.DEFAULT
        )

        cv2.imshow("Before Image", hand_image)
        cv2.imshow("Processed Image", processed_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def classify_test():
    images = get_learning_files(limit=100, shuffle=True, offset=20)

    for image_file in images:
        image_path = str(os.path.join(TRAINING_IMAGES_PATH, image_file[0]))
        image = cv2.imread(image_path)

        result, certainty = classify(algorithm=ALGORITHM.EID_SCHWENKER,
                                     payload=ImagePayload(image=image))

        cv2.imshow(f"Gesture: {result} ({certainty}%)", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def cam_test():
    camera_test(algorithm=ALGORITHM.EID_SCHWENKER, show_prediction_tresh=60)


if __name__ == "__main__":
    # process_image_test()
    classify_test()
    # cam_test()
