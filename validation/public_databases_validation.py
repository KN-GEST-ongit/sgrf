import json
import os
from datetime import datetime
from time import time

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

    iteration_start_timestamp = datetime.now().strftime("%d_%m_%YT%H:%M:%S")

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
            "timestamp": iteration_start_timestamp,
            "images_count": len(files),
            "train_files_count": len(train_files),
            "test_files_count": len(test_files),
            "algorithms": {}
        }
    
        for algorithm in algorithms:
            #skip murthy_jadon and islam_hossain as they require bg image.
            if algorithm is ALGORITHM.MURTHY_JADON or algorithm is ALGORITHM.ISLAM_HOSSAIN_ANDERSSON: continue
            print(f"Learning {algorithm.value}")

            # train validation
            train_start_time = time()
            target_model_path = str(os.path.abspath(f"./validation/trained_models/{algorithm.value}/{loader_name}/{iteration_start_timestamp}"))
            train_acc, train_loss = learn(algorithm=algorithm, learning_data=list(map(lambda file: choose_learning_data(
                algorithm=algorithm, image_path=file[0], bg_image_path=file[2], etiquette=file[1], gesture_enum=custom_options[loader]["gesture_enum"]
            ), train_files)), target_model_path=target_model_path, custom_options=custom_options[loader])
            train_end_time = time()
            train_time = train_end_time - train_start_time

            print(f"Learned {algorithm.value} with {train_acc} accuracy. Training took {train_time} seconds.")

            # test validation
            alg_correct_amount = 0
            images_amount = len(test_files)
            certainties = []
            test_start_time = time()
            predictions = []
            for image_file in test_files:
                correct_gesture = parse_etiquette(image_file[1])
                image = cv2.imread(image_file[0])
                coords = parse_file_coords(image_file[1])

                prediction, certainty = classify(algorithm=algorithm,
                                                    custom_model_dir=target_model_path,
                                                    payload=choose_payload(algorithm, None, coords, image), custom_options=custom_options[loader])
                prediction = prediction.value

                current_prediction = {
                    "correct_label": correct_gesture,
                    "predicted_label": prediction
                }
                predictions.append(current_prediction)

                if prediction == correct_gesture:
                    alg_correct_amount += 1
                certainties.append(certainty)
                
            test_end_time = time()
            test_time = test_end_time - test_start_time

            test_result = (alg_correct_amount / images_amount) * 100 if images_amount > 0 else None
            valid_certainties = [c for c in certainties if c is not None]
            test_certainty = float(round(sum(valid_certainties) / len(valid_certainties), 2)) if valid_certainties else None

            print(f"Tested algorthm {algorithm.value} on test files. Result: {test_result}%. Validation took {test_time} seconds")

            alg_result = {
                "algorithm": algorithm.value,
                "train_accuracy": train_acc,
                "train_loss": train_loss,
                "test_correct_percent": test_result,
                "test_average_certainty": test_certainty,
                "train_time": train_time,
                "test_time": test_time,
                "test_predictions": predictions     
            }
            validation_data["algorithms"][algorithm.value] = alg_result

        os.makedirs(os.path.abspath(f'./validation/results/public_databases/{iteration_start_timestamp}'), exist_ok=True)
        with open(f'./validation/results/public_databases/{iteration_start_timestamp}/{loader.__name__}_validation_results.json', 'w') as outfile:
            json.dump(validation_data, outfile, indent=2)


if __name__ == "__main__":
    algorithms = sorted(set(ALGORITHM), key=lambda a: a.value)

    for _ in range(12):
        public_databases_validation(algorithms=algorithms)
