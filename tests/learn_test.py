import os
import time

from sgrf.classifier import learn
from sgrf.data.algorithm import ALGORITHM
from scripts.choose_learning_data import choose_learning_data
from scripts.loaders import SGRFDatasetLoader


def learn_test(algorithm: ALGORITHM, images_amount: int, people_amount: int):
    # files = SGRFDatasetLoader.get_learning_files(limit=images_amount, limit_people=people_amount)
    files = SGRFDatasetLoader.get_learning_files_nextcloud(limit_people=people_amount, limit=images_amount)

    custom_options = {"verbose": 1}

    acc, loss = learn(algorithm=algorithm, learning_data=list(map(lambda file: choose_learning_data(
        algorithm=algorithm, image_path=file[0], bg_image_path=file[2], etiquette=file[1]
    ), files)), target_model_path=str(os.path.abspath(".")), custom_options=custom_options)

    return acc, loss


if __name__ == "__main__":
    # for alg in ALGORITHM:
    learn_test(ALGORITHM.MURTHY_JADON, 1000, 2)
