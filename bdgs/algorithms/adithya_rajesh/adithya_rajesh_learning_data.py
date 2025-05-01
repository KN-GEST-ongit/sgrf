from bdgs.data.gesture import GESTURE
from bdgs.models.learning_data import LearningData


class AdithyaRajeshLearningData(LearningData):
    def __init__(self, image_path: str, coords: list[tuple[int, int]], label: GESTURE):
        super().__init__(image_path, label)
        self.image_path = image_path
        self.coords = coords
        self.label = label
