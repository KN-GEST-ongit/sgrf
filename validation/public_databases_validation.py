import json
import os
from datetime import datetime

import cv2

from sgrf.classifier import learn, classify
from sgrf.data.algorithm import ALGORITHM
from scripts.choose_learning_data import choose_learning_data
from scripts.loaders import AlbarczaDatasetLoader, JochenTrieschDatasetLoader, JochenTrieschIIDatasetLoader, \
    LeapMotionDatasetLoader, NUSIIDatasetLoader, NUSDatasetLoader, ThomasMoeslundDatasetLoader, \
    SebasteinMarcelDatasetLoader
from scripts.gestures import Gesture10, Gesture12, Gesture35, Gesture6, Gesture25
from scripts.choose_payload import choose_payload
from scripts.file_coords_parser import parse_etiquette, parse_file_coords


def public_databases_validation(algorithms: set[ALGORITHM]):
    
    loaders = [AlbarczaDatasetLoader, JochenTrieschDatasetLoader, \
    JochenTrieschIIDatasetLoader, LeapMotionDatasetLoader, NUSIIDatasetLoader, NUSDatasetLoader, ThomasMoeslundDatasetLoader, \
    SebasteinMarcelDatasetLoader]

    custom_options = {
        AlbarczaDatasetLoader: { "gesture_enum": Gesture35 },
        JochenTrieschDatasetLoader: { "gesture_enum": Gesture10 },
        JochenTrieschIIDatasetLoader : { "gesture_enum": Gesture12 },
        LeapMotionDatasetLoader : { "gesture_enum": Gesture10 },
        NUSIIDatasetLoader: { "gesture_enum": Gesture10 },
        NUSDatasetLoader: { "gesture_enum": Gesture10 },
        ThomasMoeslundDatasetLoader: { "gesture_enum": Gesture25 },
        SebasteinMarcelDatasetLoader : { "gesture_enum": Gesture6 },
    }

    start_timestamp = datetime.now().strftime("%d_%m_%YT%H:%M:%S")

    for loader in loaders:
        files = loader.get_learning_files()

        loader_name = loader.__name__
        print(f"{len(files)} choosen for learning from: {loader_name}")

        split_idx = int(len(files) * 0.8)
        train_files = files[:split_idx]
        test_files  = files[split_idx:]

        print(f"Files split to 80/20 train files: {len(train_files)} and test files: {len(test_files)}")

        validation_data = {
            "loader": loader_name,
            "timestamp": start_timestamp,
            "images": len(files),
            "algorithms": {}
        }
    
        for algorithm in algorithms:
            #skip murthy_jadon and islam_hossain as they require bg image.
            if algorithm is ALGORITHM.MURTHY_JADON or algorithm is ALGORITHM.ISLAM_HOSSAIN_ANDERSSON: continue
            print(f"Learning {algorithm.value}")

            # train validation
            train_acc, train_loss = learn(algorithm=algorithm, learning_data=list(map(lambda file: choose_learning_data(
                algorithm=algorithm, image_path=file[0], bg_image_path=file[2], etiquette=file[1], gesture_enum=custom_options[loader]["gesture_enum"]
            ), train_files)), target_model_path=str(os.path.abspath(f"./validation/trained_models/{start_timestamp}")), custom_options=custom_options[loader])

            print(f"Learned {algorithm.value}: {train_acc}")

            # test validation
            alg_correct_amount = 0
            images_amount = 0
            certainties = []
            for image_file in test_files:
                images_amount += 1

                correct_gesture = parse_etiquette(image_file[1])
                image = cv2.imread(image_file[0])
                coords = parse_file_coords(image_file[1])

                prediction, certainty = classify(algorithm=algorithm,
                                                    custom_model_dir=str(os.path.abspath(f"./validation//trained_models/{start_timestamp}")),
                                                    payload=choose_payload(algorithm, None, coords, image), custom_options=custom_options[loader])
                prediction = prediction.value

                if prediction == correct_gesture:
                    alg_correct_amount += 1
                certainties.append(certainty)

            test_result = (alg_correct_amount / images_amount) * 100 if images_amount > 0 else None
            valid_certainties = [c for c in certainties if c is not None]
            test_certainty = float(round(sum(valid_certainties) / len(valid_certainties), 2)) if valid_certainties else None

            print(f"Tested algorthm {algorithm.value} on test files. Result: {test_result}%")

            alg_result = {
                "algorithm": algorithm.value,
                "train_accuracy": train_acc,
                "train_loss": train_loss,
                "test_correct_percent": test_result,
                "test_average_certainty": test_certainty
            }
            validation_data["algorithms"][algorithm.value] = alg_result

        os.makedirs(os.path.abspath(f'./validation/results/public_databases/{start_timestamp}'), exist_ok=True)
        with open(f'./validation/results/public_databases/{start_timestamp}/{loader.__name__}_validation_results.json', 'w') as outfile:
            json.dump(validation_data, outfile, indent=2)


if __name__ == "__main__":
    algorithms = set(ALGORITHM)

    for _ in range(10):
        public_databases_validation(algorithms=algorithms)
