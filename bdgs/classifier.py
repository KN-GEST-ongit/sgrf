from enum import StrEnum
from typing import Any

from cv2 import Mat
from numpy import ndarray, dtype

from bdgs.algorithms.alg_2.alg_2 import Alg2
from bdgs.algorithms.murthy_jadon.murthy_jadon import MurthyJadon
from bdgs.gesture import GESTURE


class ALGORITHM(StrEnum):
    MURTHY_JADON = "MURTHY_JADON"
    ALG_2 = "ALG_2"


ALGORITHM_FUNCTIONS = {
    ALGORITHM.MURTHY_JADON: MurthyJadon(),
    ALGORITHM.ALG_2: Alg2(),
}


def recognize(image: Mat | ndarray[Any, dtype], algorithm: ALGORITHM) -> GESTURE:
    classifier = ALGORITHM_FUNCTIONS[algorithm]
    prediction = classifier.classify(image)

    return prediction
