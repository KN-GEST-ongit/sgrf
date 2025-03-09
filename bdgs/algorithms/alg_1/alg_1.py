from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.gesture import GESTURE


class Alg1(BaseAlgorithm):
    def classify(self, image) -> GESTURE:
        return GESTURE.OK
