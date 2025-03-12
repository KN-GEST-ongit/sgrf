import os

import cv2

from bdgs import recognize
from bdgs.classifier import ALGORITHM

folder_path = "resources"

image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)

    if image is not None:
        result = recognize(image, algorithm=ALGORITHM.MURTHY_JADON)
        print(f"Result for {image_file}: {result}")
    else:
        print(f"Failed to load image: {image_file}")
