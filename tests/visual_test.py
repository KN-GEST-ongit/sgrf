import os

import cv2

from bdgs import classify
from bdgs.classifier import process_image
from bdgs.data.algorithm import ALGORITHM
from bdgs.data.processing_method import PROCESSING_METHOD
from scripts.choose_payload import choose_payload
from scripts.file_coords_parser import parse_file_coords
from scripts.get_learning_files import get_learning_files


def image_processing_visual_test(algorithm: ALGORITHM, images_amount: int):
    files = get_learning_files(shuffle=True, limit=images_amount, limit_images_in_single_person_single_recording=1,
                               base_path=os.path.abspath("../../bdgs_photos"))
    for image_file in files:
        image = cv2.imread(image_file[0])
        background = cv2.imread(image_file[2])
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


def classification_visual_test(algorithm: ALGORITHM, images_amount: int):
    files = get_learning_files(shuffle=True, limit=images_amount,
                               base_path=os.path.abspath("../../bdgs_photos"))
    for image_file in files:
        image = cv2.imread(image_file[0])
        background = cv2.imread(image_file[2])
        coords = parse_file_coords(image_file[1])
        payload = choose_payload(algorithm, background, coords, image)

        result, certainty = classify(algorithm=algorithm, payload=payload,
                                     custom_model_dir=os.path.abspath('../trained_models/sc1'))

        cv2.imshow(f"Gesture: {result} ({certainty}%)", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # image_processing_visual_test(ALGORITHM.MAUNG, 2)
    classification_visual_test(ALGORITHM.MURTHY_JADON, 5)
