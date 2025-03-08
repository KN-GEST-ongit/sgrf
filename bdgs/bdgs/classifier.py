from enum import StrEnum

from bdgs.algorithms.alg_1.alg_1 import Alg1
from bdgs.algorithms.alg_2.alg_2 import Alg2
from bdgs.gesture import GESTURE


class ALGORITHM(StrEnum):
    ALG_1 = "ALG_1"
    ALG_2 = "ALG_2"


ALGORITHM_FUNCTIONS = {
    ALGORITHM.ALG_1: Alg1(),
    ALGORITHM.ALG_2: Alg2(),
}


def recognize(image, algorithm: ALGORITHM) -> GESTURE:
    classifier = ALGORITHM_FUNCTIONS[algorithm]
    prediction = classifier.classify(image)

    return prediction
