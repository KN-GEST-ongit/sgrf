import os

import cv2

from bdgs import classify
from bdgs.algorithms.murthy_jadon.murthy_jadon_payload import MurthyJadonPayload
from bdgs.classifier import process_image
from bdgs.data.algorithm import ALGORITHM
from bdgs.data.processing_method import PROCESSING_METHOD
from scripts.common.camera_test import camera_test
from scripts.common.get_learning_files import get_learning_files
from scripts.common.vars import TRAINING_IMAGES_PATH


def process_image_test():
    images = get_learning_files()

    for image_file in images:
        image_path = str(os.path.join(TRAINING_IMAGES_PATH, image_file[0]))
        bg_image_path = str(os.path.join(TRAINING_IMAGES_PATH, image_file[2]))

        hand_image = cv2.imread(image_path)
        background_image = cv2.imread(bg_image_path)
        processed_image = process_image(
            algorithm=ALGORITHM.MURTHY_JADON,
            payload=MurthyJadonPayload(image=hand_image, bg_image=background_image),
            processing_method=PROCESSING_METHOD.DEFAULT
        )

        cv2.imshow("Processed Image", processed_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def classify_test():
    images = get_learning_files(limit=100, shuffle=True, offset=20)

    for image_file in images:
        image_path = str(os.path.join(TRAINING_IMAGES_PATH, image_file[0]))
        image = cv2.imread(image_path)
        background_image = cv2.imread(image_file[2])

        result, certainty = classify(algorithm=ALGORITHM.MURTHY_JADON,
                                     payload=MurthyJadonPayload(image=image, bg_image=background_image))

        cv2.imshow(f"Gesture: {result} ({certainty}%)", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def cam_test():
    camera_test(algorithm=ALGORITHM.MURTHY_JADON, show_prediction_tresh=70)


if __name__ == "__main__":
    # process_image_test()
    # classify_test()
    cam_test()
