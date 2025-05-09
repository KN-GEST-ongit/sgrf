import json
import os
from datetime import datetime

import cv2

from bdgs import classify
from bdgs.data.algorithm import ALGORITHM
from scripts.choose_payload import choose_payload
from scripts.file_coords_parser import parse_file_coords, parse_etiquette
from scripts.get_learning_files import get_learning_files


def classify_validation(algorithms: set[ALGORITHM], images_amount: int, repeat_amount: int = 1, people_amount: int = None):
    files = get_learning_files(shuffle=True, limit=images_amount, limit_images_in_single_person_single_recording=5,
                               limit_people=people_amount, base_path=os.path.abspath("../../bdgs_photos"))
    
    print(f"Number of files: {len(files)}")

    test_data = {
        "type": "classify_test",
        "timestamp": datetime.now().strftime("%d/%m/%YT%H:%M:%S"),
        "images": images_amount,
        "repeat_amount": repeat_amount,
        "max_people_amount": people_amount,
        "algorithms": {}
    }

    for algorithm in algorithms:
        alg_correct_amount = 0
        images_amount = 0
        certainties = []
        for i in range(repeat_amount):
            for image_file in files:
                images_amount += 1

                coords = parse_file_coords(image_file[1])
                correct_gesture = parse_etiquette(image_file[1])
                image = cv2.imread(image_file[0])
                background = cv2.imread(image_file[2])

                prediction, certainty = classify(algorithm=algorithm,
                                                 payload=choose_payload(algorithm, background, coords, image))
                prediction = prediction.value

                if prediction == correct_gesture:
                    alg_correct_amount += 1
                certainties.append(certainty)
        
        result = (alg_correct_amount / images_amount) * 100 if images_amount > 0 else 0
        alg_results = {
            "algorithm": algorithm.value,
            "correct_percent": result,
            "average_certainty": sum(certainties) / len(certainties)
        }
        test_data["algorithms"][algorithm.value] = alg_results
        
        print(f"Algorithm {algorithm}: {result}%")

    with open('results/validation_classify_results.json', 'w') as outfile:
        json.dump(test_data, outfile, indent=2)


if __name__ == "__main__":
    classify_validation(algorithms=set(ALGORITHM), images_amount=50)
