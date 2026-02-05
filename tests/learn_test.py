import os

from sgrf.classifier import learn
from sgrf.data.algorithm import ALGORITHM
from scripts.choose_learning_data import choose_learning_data
from scripts.loaders import SGRFDatasetLoader


def learn_test(algorithm: ALGORITHM, images_amount: int, people_amount: int):
    files = SGRFDatasetLoader.get_learning_files(shuffle=True, limit=images_amount,
                                                 limit_images_in_single_person_single_recording=1,
                                                 limit_people=people_amount,
                                                 base_path=os.path.abspath("../bdgs_photos"))

    custom_options = {"epochs": 20}

    acc, loss = learn(algorithm=algorithm, learning_data=list(map(lambda file: choose_learning_data(
        algorithm=algorithm, image_path=file[0], bg_image_path=file[2], etiquette=file[1]
    ), files)), target_model_path=str(os.path.abspath(".")), custom_options=custom_options)

    return acc, loss


if __name__ == "__main__":
    for alg in ALGORITHM:
        learn_test(alg, 1000, 5)
