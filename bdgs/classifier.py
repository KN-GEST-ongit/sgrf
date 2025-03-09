from enum import StrEnum
from typing import Any

from cv2 import Mat
from numpy import ndarray, dtype

from bdgs.algorithms.alg_1.alg_1 import Alg1
from bdgs.algorithms.alg_2.alg_2 import Alg2
from bdgs.gesture import GESTURE


class ALGORITHM(StrEnum):
    ALG_1 = "ALG_1"
    ALG_2 = "ALG_2"  # added


ALGORITHM_FUNCTIONS = {
    ALGORITHM.ALG_1: Alg1(),
    ALGORITHM.ALG_2: Alg2(),  # added
}


def recognize(image: Mat | ndarray[Any, dtype], algorithm: ALGORITHM) -> GESTURE:
    classifier = ALGORITHM_FUNCTIONS[algorithm]
    prediction = classifier.classify(image)

    return prediction
