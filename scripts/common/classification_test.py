import os

import cv2

from bdgs import classify
from bdgs.data.algorithm import ALGORITHM
from scripts.common.crop_image import parse_file_coords
from scripts.common.get_learning_files import get_learning_files
from scripts.common.vars import TRAINING_IMAGES_PATH
from validation.learning_test import choose_payload


def classification_test(algorithm: ALGORITHM):
    images = get_learning_files(limit=100, shuffle=True, offset=20, limit_images_in_single_person_single_recording=1,
                                limit_recordings_of_single_person_single_gesture=1)

    for image_file in images:
        image_path = str(os.path.join(TRAINING_IMAGES_PATH, image_file[0]))
        image = cv2.imread(image_path)
        background = cv2.imread(image_file[2])

        coords = parse_file_coords(image_file[1])

        payload = choose_payload(algorithm, background, coords, image)

        result, certainty = classify(algorithm=algorithm, payload=payload)

        cv2.imshow(f"Gesture: {result} ({certainty}%)", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
