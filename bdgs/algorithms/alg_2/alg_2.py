from numpy import ndarray

from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.gesture import GESTURE


class Alg2(BaseAlgorithm):
    def process_image(self, image: ndarray) -> ndarray:
        pass

    def classify(self, image) -> GESTURE:
        return GESTURE.HELLO
