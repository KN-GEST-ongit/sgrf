import json
import os
from datetime import datetime

from bdgs.classifier import learn
from bdgs.data.algorithm import ALGORITHM
from scripts.choose_learning_data import choose_learning_data
from scripts.get_learning_files import get_learning_files


def learn_validation(algorithms: set[ALGORITHM], people_amount: int = None, images_amount: int = None,
                     limit_images_in_single_person_single_recording=None,
                     limit_recordings_of_single_person_single_gesture=None):
    files = get_learning_files(shuffle=True, limit=images_amount,
                               limit_images_in_single_person_single_recording=limit_images_in_single_person_single_recording,
                               limit_recordings_of_single_person_single_gesture=limit_recordings_of_single_person_single_gesture,
                               limit_people=people_amount, base_path=os.path.abspath("../../bdgs_photos"))

    print(f"{len(files)} choosen for learning")

    test_data = {
        "type": "learn_test",
        "timestamp": datetime.now().strftime("%d/%m/%YT%H:%M:%S"),
        "images": len(files),
        "max_people_amount": people_amount,
        "algorithms": {}
    }

    for algorithm in algorithms:
        acc, loss = learn(algorithm=algorithm, learning_data=list(map(lambda file: choose_learning_data(
            algorithm=algorithm, image_path=file[0], bg_image_path=file[2], etiquette=file[1]
        ), files)), target_model_path=str(os.path.abspath("../trained_models")))

        alg_results = {
            "algorithm": algorithm.value,
            "accuracy": acc,
            "loss": loss,
        }
        test_data["algorithms"][algorithm.value] = alg_results

        print(f"Learned {algorithm.value}: {acc}")

    with open('results/validation_learn_results.json', 'w') as outfile:
        json.dump(test_data, outfile, indent=2)


if __name__ == "__main__":
    learn_validation(algorithms=set(ALGORITHM)) #scenario 1
    # learn_validation(algorithms=set(ALGORITHM), limit_recordings_of_single_person_single_gesture=2) #scenario 2
    # learn_validation(algorithms=set(ALGORITHM), limit_images_in_single_person_single_recording=10) #scenario 3
    # learn_validation(algorithms=set(ALGORITHM), people_amount=2) #scenario 4
