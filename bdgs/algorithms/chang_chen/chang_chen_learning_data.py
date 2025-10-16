from bdgs.data.gesture import GESTURE
from bdgs.models.learning_data import LearningData


class ChangChenLearningData(LearningData):
    def __init__(self, image_path: str, label: GESTURE):
        super().__init__(image_path, label)
    