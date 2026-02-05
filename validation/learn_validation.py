import json
import os
from datetime import datetime

from sgrf.classifier import learn
from sgrf.data.algorithm import ALGORITHM
from scripts.choose_learning_data import choose_learning_data
from scripts.loaders import SGRFDatasetLoader


def learn_validation(scenario_name: str, algorithms: set[ALGORITHM], people_amount: int = None,
                     images_amount: int = None,
                     limit_images_in_single_person_single_recording=None,
                     limit_recordings_of_single_person_single_gesture=None):
    files = SGRFDatasetLoader.get_learning_files(shuffle=True, limit=images_amount,
                                                 limit_images_in_single_person_single_recording=limit_images_in_single_person_single_recording,
                                                 limit_recordings_of_single_person_single_gesture=limit_recordings_of_single_person_single_gesture,
                                                 limit_people=people_amount,
                                                 base_path=os.path.abspath("../../bdgs_photos"))

    print(f"{len(files)} choosen for learning")

    test_data = {
        "type": "learn_test",
        "timestamp": datetime.now().strftime("%d/%m/%YT%H:%M:%S"),
        "images": len(files),
        "max_people_amount": people_amount,
        "limit_images_in_single_person_single_recording": limit_images_in_single_person_single_recording,
        "limit_recordings_of_single_person_single_gesture": limit_recordings_of_single_person_single_gesture,
        "algorithms": {}
    }

    for algorithm in algorithms:
        print(f"Learning {algorithm.value}")

        acc, loss = learn(algorithm=algorithm, learning_data=list(map(lambda file: choose_learning_data(
            algorithm=algorithm, image_path=file[0], bg_image_path=file[2], etiquette=file[1]
        ), files)), target_model_path=str(os.path.abspath(f"../trained_models/{scenario_name}")))

        alg_results = {
            "algorithm": algorithm.value,
            "accuracy": acc,
            "loss": loss,
        }
        test_data["algorithms"][algorithm.value] = alg_results

        print(f"Learned {algorithm.value}: {acc}")

    os.makedirs(os.path.abspath(f'results/{scenario_name}'), exist_ok=True)
    with open(f'results/{scenario_name}/{scenario_name}_validation_learn_results.json', 'w') as outfile:
        json.dump(test_data, outfile, indent=2)


if __name__ == "__main__":
    # algorithms = set(ALGORITHM)
    algorithms = {ALGORITHM.CHANG_CHEN, ALGORITHM.EID_SCHWENKER, ALGORITHM.GUPTA_JAAFAR, ALGORITHM.JOSHI_KUMAR,
                  ALGORITHM.MAUNG, ALGORITHM.MOHMMAD_DADI, ALGORITHM.MURTHY_JADON, ALGORITHM.NGUYEN_HUYNH,
                  ALGORITHM.PINTO_BORGES}

    learn_validation("sc1", algorithms=algorithms)  # scenario 1
    # learn_validation("sc2", algorithms=algorithms, limit_recordings_of_single_person_single_gesture=2) #scenario 2
    # learn_validation("sc3", algorithms=algorithms, limit_images_in_single_person_single_recording=10)  # scenario 3
    # learn_validation("sc4", algorithms=algorithms, people_amount=2)  # scenario 4
