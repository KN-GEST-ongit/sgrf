import os

import cv2
import numpy as np

from bdgs import recognize
from bdgs.algorithms.adithya_rajesh.adithya_rajesh import AdithyaRajesh
from bdgs.models.image_payload import ImagePayload
from scripts.common.get_learning_files import get_learning_files

folder_path = os.path.abspath("../bdgs_photos")

print(folder_path)

def test_process_image():
    image_files = get_learning_files()

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file[0])
        image = cv2.imread(image_path)

        if image is not None:
            #extract hand
            image_details = image_file[1]
            image_details = image_details.replace("(", "").replace(")", "")
            hand_details = list(map(int, image_details.split(" ")))

            image_label = hand_details[0]
            corner1 = (hand_details[1], hand_details[2])
            corner2 = (hand_details[3], hand_details[4])
            
            #crop the hand
            image = image[corner1[1]:corner2[1], corner1[0]:corner2[0]]

            alg = AdithyaRajesh()
            payload = ImagePayload(image)

            processed_image = alg.process_image(payload)
            print("Processed image shape for model: ", processed_image.shape)

            #remove batch dimension (1, 100, 100, 3) -> (100, 100, 3)
            image_without_batch = np.squeeze(processed_image)
            #go back to BGR from RGB
            image_without_batch = cv2.cvtColor(image_without_batch, cv2.COLOR_RGB2BGR)
            #set array datatype back to uint8
            image_without_batch = np.astype(image_without_batch, np.uint8)

            print("Processed image shape for opencv: ", image_without_batch.shape)

            print(f"Image label: {image_label}")
            cv2.imshow("image", image_without_batch)
            cv2.waitKey(2000)

        else:
            print(f"Failed to load image: {image_file}")

test_process_image()
