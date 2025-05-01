from bdgs.data.gesture import GESTURE


class LearningData:
    def __init__(self, image_path: str, label: GESTURE):
        self.image_path = image_path
        self.label = label
