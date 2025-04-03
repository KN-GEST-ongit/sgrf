import os
import random

folder_path = os.path.abspath("../../../bdgs_photos")


def get_learning_files():
    image_files = []
    classify_file = None
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".txt"):
                classify_file = os.path.join(root, file)
                break

        if classify_file is None or len(files) == 0: continue

        with open(classify_file, "r") as f:
            classify_row = [line.split("\n")[0] for line in f]
        files.pop(0)  # remove classify file

        for index in range(len(files) - 1):
            if files[index].lower().endswith(('.png', '.jpg', '.jpeg')) and classify_row[index].split(" ")[0] != "0":
                image_files.append((os.path.join(root, files[index]), classify_row[index]))

    random.shuffle(image_files)
    return image_files
