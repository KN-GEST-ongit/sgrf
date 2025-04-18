import os

import cv2

from bdgs.algorithms.islam_hossain_andersson.islam_hossain_andersson import IslamHossainAndersson
from bdgs.data.gesture import GESTURE
from bdgs.data.algorithm import ALGORITHM
from bdgs.algorithms.islam_hossain_andersson.islam_hossain_andersson_payload import IslamHossainAnderssonPayload
from scripts.common.camera_test import camera_test
from scripts.common.crop_image import crop_image
from scripts.common.get_learning_files import get_learning_files
from scripts.common.vars import TRAINING_IMAGES_PATH


def test_process_image():
    image_files = get_learning_files()

    for image_file in image_files:
        image_path = os.path.join(TRAINING_IMAGES_PATH, image_file[0])
        bg_image_path = str(os.path.join(TRAINING_IMAGES_PATH, image_file[2]))
        background = cv2.imread(str(bg_image_path))
        image = cv2.imread(str(image_path))

        if image is not None:
            image_label = int(image_file[1].split(" ")[0])
            image = crop_image(image, image_file[1])
            background = crop_image(background, image_file[1])
            alg = IslamHossainAndersson()
            payload = IslamHossainAnderssonPayload(image=image, bg_image=background)
            processed_image = alg.process_image(payload)

            print(f"Image label: {image_label}")
            cv2.imshow("image", processed_image)
            cv2.waitKey(2000)
        else:
            print(f"Failed to load image: {image_file}")

def classify_test():
    image_files = get_learning_files()

    for image_file in image_files:
        image_path = os.path.join(TRAINING_IMAGES_PATH, image_file[0])
        image = cv2.imread(str(image_path))
        bg_image_path = os.path.join(TRAINING_IMAGES_PATH, image_file[2])
        bg_image = crop_image(cv2.imread(str(bg_image_path)), image_file[1])

        if image is not None:
            image_label = int(image_file[1].split(" ")[0])
            image = crop_image(image, image_file[1])

            alg = IslamHossainAndersson()
            payload = IslamHossainAnderssonPayload(image=image, bg_image=bg_image)
            predicted_class, certainty = alg.classify(payload=payload)
            print(f"Correct class: {GESTURE(image_label).name}")
            print(f"Predicted class: {predicted_class}, certainty: {certainty}%")
            cv2.imshow("image", image)
            cv2.waitKey(2000)
        else:
            print(f"Failed to load image: {image_file}")

def cam_test():
    camera_test(algorithm=ALGORITHM.ISLAM_HOSSAIN_ANDERSSON, show_prediction_tresh=70)

if __name__ == "__main__":
    #test_process_image()
    classify_test()
    #cam_test()
