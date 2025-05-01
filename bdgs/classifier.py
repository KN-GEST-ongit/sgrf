from numpy import ndarray

from bdgs.data.algorithm import ALGORITHM
from bdgs.data.algorithm_functions import ALGORITHM_FUNCTIONS
from bdgs.data.gesture import GESTURE
from bdgs.data.processing_method import PROCESSING_METHOD
from bdgs.models.image_payload import ImagePayload
from bdgs.models.learning_data import LearningData


def process_image(algorithm: ALGORITHM, payload: ImagePayload,
                  processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> ndarray:
    classifier = ALGORITHM_FUNCTIONS[algorithm]
    processed = classifier.process_image(payload, processing_method)

    return processed


def classify(algorithm: ALGORITHM, payload: ImagePayload,
             processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> (GESTURE, int):
    classifier = ALGORITHM_FUNCTIONS[algorithm]
    prediction, certainty = classifier.classify(payload, processing_method)

    return prediction, certainty


def learn(algorithm: ALGORITHM, learning_data: list[LearningData], target_model_path: str):
    classifier = ALGORITHM_FUNCTIONS[algorithm]
    acc, loss = classifier.learn(learning_data, target_model_path)

    return acc, loss
