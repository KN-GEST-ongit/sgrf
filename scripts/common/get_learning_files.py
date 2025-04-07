import os
import random

from scripts.common.vars import TRAINING_IMAGES_PATH


def get_learning_files(skip_empty=True, shuffle=True, limit=None, offset=0):
    image_files = []
    classify_file = None
    for root, _, files in os.walk(TRAINING_IMAGES_PATH):
        for file in files:
            if file.lower().endswith(".txt"):
                classify_file = os.path.join(root, file)
                break
        if classify_file is None or len(files) == 0: continue
        with open(classify_file, "r") as f:
            classify_row = [line.split("\n")[0] for line in f]
        files.pop(0)
        bg_image = files[0]

        files = sorted(files)

        for index in range(len(files) - 1):
            if files[index].lower().endswith(('.png', '.jpg', '.jpeg')):
                if skip_empty:
                    if classify_row[index].split(" ")[0] != "0":
                        image_files.append(
                            (os.path.join(root, files[index]), classify_row[index], (os.path.join(root, bg_image))))
                else:
                    image_files.append(
                        (os.path.join(root, files[index]), classify_row[index], (os.path.join(root, bg_image))))

    if shuffle: random.shuffle(image_files)
    return image_files[offset:(limit + offset if limit is not None else None)]
