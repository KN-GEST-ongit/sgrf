from bdgs.data.gesture import GESTURE
from bdgs.models.learning_data import LearningData


class MohantyRambhatlaLearningData(LearningData):
    def __init__(self, image_path: str, label: GESTURE, coords: list[tuple[int, int]]):
        super().__init__(image_path, label)
        self.coords = coords
    