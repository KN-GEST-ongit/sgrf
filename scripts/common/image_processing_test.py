import os

import cv2

from bdgs.algorithms.islam_hossain_andersson.islam_hossain_andersson_payload import IslamHossainAnderssonPayload
from bdgs.algorithms.murthy_jadon.murthy_jadon_payload import MurthyJadonPayload
from bdgs.algorithms.pinto_borges.pinto_borges_payload import PintoBorgesPayload
from bdgs.algorithms.adithya_rajesh.adithya_rajesh_payload import AdithyaRajeshPayload
from bdgs.classifier import process_image
from bdgs.data.algorithm import ALGORITHM
from bdgs.data.processing_method import PROCESSING_METHOD
from bdgs.models.image_payload import ImagePayload
from scripts.common.crop_image import parse_file_coords
from scripts.common.get_learning_files import get_learning_files
from scripts.common.vars import TRAINING_IMAGES_PATH


def image_processing_test(algorithm: ALGORITHM):
    images = get_learning_files(shuffle=False, limit_recordings_of_single_person_single_gesture=2,
                                limit_images_in_single_person_single_recording=1)

    for image_file in images:
        image_path = str(os.path.join(TRAINING_IMAGES_PATH, image_file[0]))
        bg_image_path = str(os.path.join(TRAINING_IMAGES_PATH, image_file[2]))

        image = cv2.imread(image_path)
        background = cv2.imread(bg_image_path)
        coords = parse_file_coords(image_file[1])

        if algorithm == ALGORITHM.MURTHY_JADON:
            payload = MurthyJadonPayload(image=image, bg_image=background)
        elif algorithm == ALGORITHM.ISLAM_HOSSAIN_ANDERSSON:
            payload = IslamHossainAnderssonPayload(image=image, bg_image=background, coords=coords)
        elif algorithm == ALGORITHM.PINTO_BORGES:
            payload = PintoBorgesPayload(image=image, coords=coords)
        elif algorithm == ALGORITHM.ADITHYA_RAJESH:
            payload = AdithyaRajeshPayload(image=image, coords=coords)
        else:
            payload = ImagePayload(image=image)

        processed_image = process_image(
            algorithm=algorithm,
            payload=payload,
            processing_method=PROCESSING_METHOD.DEFAULT
        )

        cv2.imshow("Before Image", image)
        cv2.imshow("Processed Image", processed_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
