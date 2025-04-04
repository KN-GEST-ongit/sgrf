from enum import StrEnum

from numpy import ndarray

from bdgs.algorithms.adithya_rajesh.adithya_rajesh import AdithyaRajesh
from bdgs.algorithms.maung.maung import Maung
from bdgs.algorithms.murthy_jadon.murthy_jadon import MurthyJadon
from bdgs.gesture import GESTURE
from bdgs.models.image_payload import ImagePayload


class ALGORITHM(StrEnum):
    MURTHY_JADON = "MURTHY_JADON"
    MAUNG = "MAUNG"
    ADITHYA_RAJESH = "ADITHYA_RAJESH"


ALGORITHM_FUNCTIONS = {
    ALGORITHM.MURTHY_JADON: MurthyJadon(),
    ALGORITHM.MAUNG: Maung(),
    ALGORITHM.ADITHYA_RAJESH: AdithyaRajesh(),
}


def process_image(payload: ImagePayload, algorithm: ALGORITHM) -> ndarray:
    classifier = ALGORITHM_FUNCTIONS[algorithm]
    processed = classifier.process_image(payload)

    return processed


def classify(payload: ImagePayload, algorithm: ALGORITHM) -> GESTURE:
    classifier = ALGORITHM_FUNCTIONS[algorithm]
    prediction = classifier.classify(payload)

    return prediction
