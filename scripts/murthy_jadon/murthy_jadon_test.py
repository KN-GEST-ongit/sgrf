import os

import cv2

from bdgs import recognize
from bdgs.algorithms.murthy_jadon.murthy_jadon import MurthyJadon
from bdgs.algorithms.murthy_jadon.murthy_jadon_payload import MurthyJadonPayload
from bdgs.classifier import ALGORITHM
from scripts.common.get_learning_files import get_learning_files

folder_path = os.path.abspath("../../../bdgs_photos")


def test_process_image():
    images = get_learning_files()

    for image_file in images:
        image_path = os.path.join(folder_path, image_file[0])
        print(image_path)
        image = cv2.imread(image_path)

        if image is not None:
            cv2.imshow("Before", image)
            cv2.waitKey(0)

            alg = MurthyJadon()
            alg_payload = MurthyJadonPayload(image, image)
            processed_image = alg.process_image(alg_payload)

            cv2.imshow("After", processed_image)
            cv2.waitKey(0)

            cv2.destroyAllWindows()

            result = recognize(image, algorithm=ALGORITHM.MURTHY_JADON)
            print(f"Result for {image_file}: {result}")
        else:
            print(f"Failed to load image: {image_file}")


# def test_classify():
#     for image_file in image_files:
#         image_path = os.path.join(folder_path, image_file)
#         image = cv2.imread(image_path)
# 
#         if image is not None:
#             result = recognize(image, algorithm=ALGORITHM.MURTHY_JADON)
#             print(f"Result for {image_file}: {result}")
#         else:
#             print(f"Failed to load image: {image_file}")


test_process_image()
# test_classify()
