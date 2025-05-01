import json
import os
from datetime import datetime

from bdgs.classifier import learn
from bdgs.data.algorithm import ALGORITHM
from scripts.choose_learning_data import choose_learning_data
from scripts.get_learning_files import get_learning_files


def learn_test(algorithms: set[ALGORITHM], images_amount: int, people_amount: int):
    files = get_learning_files(shuffle=True, limit=images_amount, limit_images_in_single_person_single_recording=1,
                               limit_people=people_amount, base_path=os.path.abspath("../../bdgs_photos"))

    test_data = {
        "type": "learn_test",
        "timestamp": datetime.now().strftime("%d/%m/%YT%H:%M:%S"),
        "images": images_amount,
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

    with open('results/validation_learn_results.json', 'w') as outfile:
        json.dump(test_data, outfile, indent=2)


if __name__ == "__main__":
    learn_test(algorithms=set(ALGORITHM), images_amount=1000, people_amount=5)
