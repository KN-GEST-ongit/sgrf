import os

from bdgs.classifier import learn
from bdgs.data.algorithm import ALGORITHM
from scripts.choose_learning_data import choose_learning_data
from scripts.get_learning_files import get_learning_files


def learn_test(algorithm: ALGORITHM, images_amount: int, people_amount: int):
    files = get_learning_files(shuffle=True, limit=images_amount, limit_images_in_single_person_single_recording=1,
                               limit_people=people_amount, base_path=os.path.abspath("../../bdgs_photos"))

    acc, loss = learn(algorithm=algorithm, learning_data=list(map(lambda file: choose_learning_data(
        algorithm=algorithm, image_path=file[0], bg_image_path=file[2], etiquette=file[1]
    ), files)), target_model_path=str(os.path.abspath(".")))

    return acc, loss


if __name__ == "__main__":
    learn_test(ALGORITHM.CHANG_CHEN, 1000, 5)
