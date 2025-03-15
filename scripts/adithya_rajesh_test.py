import os

import cv2
import numpy as np

from bdgs import recognize
from bdgs.algorithms.adithya_rajesh.adithya_rajesh import AdithyaRajesh

folder_path = "../resources"

image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


def test_process_image():
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)

        alg = AdithyaRajesh()
        if image is not None:
            processed_image = alg.process_image(image)
            print("Processed image shape for model: ", processed_image.shape)

            #remove batch dimension (1, 100, 100, 3) -> (100, 100, 3)
            image_without_batch = np.squeeze(processed_image)
            #go back to BGR from RGB
            image_without_batch = cv2.cvtColor(image_without_batch, cv2.COLOR_RGB2BGR)
            #set array datatype back to uint8
            image_without_batch = np.astype(image_without_batch, np.uint8)

            print("Processed image shape for opencv: ", image_without_batch.shape)

            cv2.imshow("image", image_without_batch)
            cv2.waitKey(0)

        else:
            print(f"Failed to load image: {image_file}")

test_process_image()
