import os

import cv2

from bdgs import classify
from bdgs.algorithms.maung.maung import Maung
from bdgs.data.algorithm import ALGORITHM
from bdgs.models.image_payload import ImagePayload
from scripts.common.crop_image import crop_image
from scripts.common.get_learning_files import get_learning_files
from scripts.common.vars import TRAINING_IMAGES_PATH


def test_process_image():
    images = get_learning_files()

    for img, coords, bg in images:
        image_path = str(os.path.join(TRAINING_IMAGES_PATH, img))
        print(image_path)
        image = cv2.imread(image_path)

        image = crop_image(image, coords)

        if image is not None:
            cv2.imshow("Before", image)
            cv2.waitKey(0)

            alg = Maung()
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
