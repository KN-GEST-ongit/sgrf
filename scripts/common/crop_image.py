import numpy as np


def crop_image(image: np.ndarray, coords: list[tuple[int, int]]) -> np.ndarray:
    (x1, y1), (x2, y2) = coords

    cropped_image = image[y1:y2, x1:x2]

    return cropped_image


def parse_file_coords(coord_string: str) -> list[tuple[int, int]]:
    parts = coord_string.split(" ", 1)[1]
    coords = parts.split(") (")

    result = []
    for coord in coords[:2]:
        coord = coord.strip("()")
        x, y = map(int, coord.split())
        result.append((x, y))

    return result
