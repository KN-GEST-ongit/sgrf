from enum import StrEnum

from bdgs.algorithms.alg_1.alg_1 import alg_1
from bdgs.algorithms.alg_2.alg_2 import alg_2
from bdgs.gesture import Gesture


class ALGORITHM(StrEnum):
    ALG_1 = "ALG_1"
    ALG_2 = "ALG_2"


ALGORITHM_FUNCTIONS = {
    ALGORITHM.ALG_1: alg_1,
    ALGORITHM.ALG_2: alg_2,
}


def recognize(image, algorithm: ALGORITHM) -> Gesture:
    classifier = ALGORITHM_FUNCTIONS[algorithm]
    prediction = classifier(image)

    return prediction
