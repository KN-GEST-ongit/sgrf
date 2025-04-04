import numpy as np


def crop_image(image: np.ndarray, coords: str) -> np.ndarray:
    coordinates = coords.split(" ", 1)[1]
    x1, y1 = map(int, coordinates.split(") (")[0].strip("()").split())
    x2, y2 = map(int, coordinates.split(") (")[1].strip("()").split())

    image = image[y1:y2, x1:x2]

    return image
