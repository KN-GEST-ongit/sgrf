import numpy as np


def crop_image(image: np.ndarray, coords: list[tuple[int, int]]) -> np.ndarray:
    (x1, y1), (x2, y2) = coords

    cropped_image = image[y1:y2, x1:x2]

    return cropped_image
