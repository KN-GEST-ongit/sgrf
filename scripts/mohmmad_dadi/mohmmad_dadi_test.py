import os

import cv2

from bdgs import classify
from bdgs.algorithms.mohmmad_dadi.mohmmad_dadi import MohmmadDadi
from bdgs.data.algorithm import ALGORITHM
from bdgs.models.image_payload import ImagePayload
from scripts.common.camera_test import camera_test
from scripts.common.get_learning_files import get_learning_files
from scripts.common.vars import TRAINING_IMAGES_PATH


def test_process_image():
    images = get_learning_files()

    for img, coords, bg in images:
        image_path = str(os.path.join(TRAINING_IMAGES_PATH, img))
        print(image_path)
        image = cv2.imread(image_path)

        if image is not None:
            cv2.imshow("Before", image)
            cv2.waitKey(0)

            alg = MohmmadDadi()
            alg_payload = ImagePayload(image)
            processed_image = alg.process_image(alg_payload)
            cv2.imshow("After", processed_image)

            # processed_image = alg.process_image(alg_payload, processing_method=PROCESSING_METHOD.ADITHYA_RAJESH)
            # cv2.imshow("After", processed_image[0])
            cv2.waitKey(0)

            cv2.destroyAllWindows()

            result = classify(payload=ImagePayload(image), algorithm=ALGORITHM.MAUNG)
            print(f"Result for {img}: {result}")
        else:
            print(f"Failed to load image: {img}")


def classify_test():
    images = get_learning_files(limit=100, shuffle=True, offset=20)

    for image, hand_recognition_data, _ in images:
        image_path = str(os.path.join(TRAINING_IMAGES_PATH, image))
        hand_image = cv2.imread(image_path)
        result, certainty = classify(algorithm=ALGORITHM.MOHMMAD_DADI,
                                     payload=ImagePayload(image=hand_image))

        cv2.imshow(f"Gesture: {result} ({certainty}%)", hand_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def cam_test():
    camera_test(algorithm=ALGORITHM.MOHMMAD_DADI, show_prediction_tresh=60)


# test_process_image()
classify_test()
# cam_test()
