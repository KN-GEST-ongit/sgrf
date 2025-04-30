import json
import os
from datetime import datetime

import cv2

from bdgs import classify
from bdgs.algorithms.adithya_rajesh.adithya_rajesh_payload import AdithyaRajeshPayload
from bdgs.algorithms.islam_hossain_andersson.islam_hossain_andersson_payload import IslamHossainAnderssonPayload
from bdgs.algorithms.murthy_jadon.murthy_jadon_payload import MurthyJadonPayload
from bdgs.algorithms.pinto_borges.pinto_borges_payload import PintoBorgesPayload
from bdgs.data.algorithm import ALGORITHM
from bdgs.models.image_payload import ImagePayload
from scripts.common.crop_image import parse_file_coords, parse_etiquette
from scripts.common.get_learning_files import get_learning_files


def classify_test(algorithms: set[ALGORITHM], images_amount: int, repeat_amount: int, people_amount: int):
    files = get_learning_files(shuffle=True, limit=images_amount, limit_images_in_single_person_single_recording=1,
                               limit_people=people_amount, base_path=os.path.abspath("../../bdgs_photos"))

    test_data = {
        "timestamp": datetime.now().strftime("%d/%m/%YT%H:%M:%S"),
        "images": images_amount,
        "repeat_amount": repeat_amount,
        "max_people_amount": people_amount,
        "algorithms": {}
    }

    for algorithm in algorithms:
        alg_results = {
            "algorithm": algorithm.value,
            "corrent_percent": 0,
            "average_certainty": 0.0,
        }
        alg_correct_amount = 0
        images_amount = 0
        certainties = []
        for i in range(repeat_amount):
            for image_file in files:
                images_amount += 1
                image_path = str(os.path.join("../../bdgs_photos", image_file[0]))
                coords = parse_file_coords(image_file[1])
                correct_gesture = parse_etiquette(image_file[1]) + 1
                bg_image_path = str(os.path.join("../../bdgs_photos", image_file[2]))

                image = cv2.imread(image_path)
                background = cv2.imread(bg_image_path)

                prediction, certainty = classify(algorithm=algorithm,
                                                 payload=choose_payload(algorithm, background, coords, image))
                prediction = prediction.value

                if prediction == correct_gesture:
                    alg_correct_amount += 1
                certainties.append(certainty)

        alg_results["corrent_percent"] = (alg_correct_amount / (images_amount * repeat_amount)) * 100
        alg_results["average_certainty"] = sum(certainties) / len(certainties)
        test_data["algorithms"][algorithm.value] = alg_results

    with open('validation_results.json', 'w') as outfile:
        json.dump(test_data, outfile, indent=2)


def choose_payload(algorithm, background, coords, image):
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
    return payload


if __name__ == "__main__":
    classify_test(algorithms=set(ALGORITHM), images_amount=10, repeat_amount=2, people_amount=5)
