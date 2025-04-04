import os

import cv2

from bdgs import classify
from bdgs.algorithms.murthy_jadon.murthy_jadon_payload import MurthyJadonPayload
from bdgs.classifier import ALGORITHM, process_image
from scripts.common.get_learning_files import get_learning_files
from scripts.common.vars import training_images_path


def process_image_test():
    images = get_learning_files()

    for image_file in images:
        image_path = str(os.path.join(training_images_path, image_file[0]))
        bg_image_path = str(os.path.join(training_images_path, image_file[2]))

        hand_image = cv2.imread(image_path)
        background_image = cv2.imread(bg_image_path)
        processed_image = process_image(
            algorithm=ALGORITHM.MURTHY_JADON,
            payload=MurthyJadonPayload(image=hand_image, bg_image=background_image)
        )

        cv2.imshow("Processed Image", processed_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def classify_test():
    images = get_learning_files(limit=100, shuffle=True, offset=20)

    for image_file in images:
        image_path = str(os.path.join(training_images_path, image_file[0]))
        image = cv2.imread(image_path)
        background_image = cv2.imread(image_file[2])

        result = classify(algorithm=ALGORITHM.MURTHY_JADON,
                          payload=MurthyJadonPayload(image=image, bg_image=background_image))

        cv2.imshow(f"Gesture: {result}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # process_image_test()
    classify_test()
