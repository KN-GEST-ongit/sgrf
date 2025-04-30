import os

import cv2

from bdgs.classifier import process_image
from bdgs.data.algorithm import ALGORITHM
from bdgs.data.processing_method import PROCESSING_METHOD
from scripts.common.crop_image import parse_file_coords
from scripts.common.get_learning_files import get_learning_files
from scripts.common.vars import TRAINING_IMAGES_PATH
from validation.learning_test import choose_payload


def image_processing_test(algorithm: ALGORITHM):
    images = get_learning_files(shuffle=False,
                                limit_images_in_single_person_single_recording=1, limit_people=2,
                                limit_recordings_of_single_person_single_gesture=1)

    for image_file in images:
        image_path = str(os.path.join(TRAINING_IMAGES_PATH, image_file[0]))
        bg_image_path = str(os.path.join(TRAINING_IMAGES_PATH, image_file[2]))

        image = cv2.imread(image_path)
        background = cv2.imread(bg_image_path)
        coords = parse_file_coords(image_file[1])

        payload = choose_payload(algorithm, background, coords, image)

        processed_image = process_image(
            algorithm=algorithm,
            payload=payload,
            processing_method=PROCESSING_METHOD.DEFAULT
        )

        cv2.imshow("Before Image", image)
        cv2.imshow("Processed Image", processed_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
