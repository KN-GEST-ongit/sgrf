import os

import cv2

from bdgs import recognize
from bdgs.algorithms.maung.maung import Maung
from bdgs.classifier import ALGORITHM
from bdgs.models.image_payload import ImagePayload
from scripts.common.get_learning_files import get_learning_files

folder_path = os.path.abspath("../../../bdgs_photos")


def test_process_image():
    images = get_learning_files()

    for image_file in images:
        image_path = os.path.join(folder_path, image_file[0])
        print(image_path)
        image = cv2.imread(image_path)

        coords = image_file[1].split(" ", 1)[1]
        x1, y1 = map(int, coords.split(") (")[0].strip("()").split())
        x2, y2 = map(int, coords.split(") (")[1].strip("()").split())

        image = image[y1:y2, x1:x2]

        if image is not None:
            cv2.imshow("Before", image)
            cv2.waitKey(0)

            alg = Maung()
            alg_payload = ImagePayload(image)
            processed_image = alg.process_image(alg_payload)

            cv2.imshow("After", processed_image)
            cv2.waitKey(0)

            cv2.destroyAllWindows()

            result = recognize(image, algorithm=ALGORITHM.MAUNG)
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
