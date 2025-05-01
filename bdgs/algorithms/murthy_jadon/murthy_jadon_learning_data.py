from bdgs.data.gesture import GESTURE
from bdgs.models.learning_data import LearningData


class MurthyJadonLearningData(LearningData):
    def __init__(self, image_path: str, bg_image_path: str, label: GESTURE):
        super().__init__(image_path, label)
        self.image_path = image_path
        self.bg_image_path = bg_image_path
        self.label = label
