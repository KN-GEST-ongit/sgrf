import os

from scripts.vars import ALGORITHMS_CONFIG_DATA_PATH

ALGORITHM_FILENAME = "algorithm.py"
ALGORITHM_FUNCTIONS_FILENAME = "algorithm_functions.py"
PROCESSING_METHOD_FILENAME = "processing_method.py"
ALGORITHM_BASE_PATH = "../bdgs/algorithms"
PAYLOAD_SCRIPTS_FILE = "choose_payload.py"
LEARNING_DATA_SCRIPTS_FILE \
    = "choose_learning_data.py"


def add_algorithm():
    while True:
        algorithm_name = input("Algorithm name: ").strip().upper()
        if not algorithm_name:
            print("Invalid name.")
            continue

        module_name = algorithm_name.lower()
        class_name = ''.join(word.capitalize() for word in algorithm_name.lower().split('_'))
        modify_payload = False
        modify_learning_data = False

        custom_payload = input("Do you want to add custom payload class? (Y/n) ").strip()
        if custom_payload.lower() == "y":
            add_custom_payload(module_name, class_name)
            modify_payload = input(
                "Do you want to add your custom payload class to scripts choices section? (Y/n) ").strip().lower() == "y"

        custom_learning_data = input("Do you want to add custom learning data class? (Y/n) ").strip()
        if custom_learning_data.lower() == "y":
            add_custom_learning_data(module_name, class_name)
            modify_learning_data = input(
                "Do you want to add your custom learning data class to scripts choices section? (Y/n) ").strip().lower() == "y"

        modify_algorithm_enum(algorithm_name)
        modify_functions(algorithm_name, class_name, module_name)
        create_algorithm_skeleton(module_name, class_name)

        if modify_payload:
            modify_payload_script(module_name, class_name, algorithm_name)
        if modify_learning_data:
            modify_learning_data_script(module_name, class_name, algorithm_name)

        print(f"Algorithm added: {algorithm_name}")
        return


def modify_algorithm_enum(algorithm_name):
    algorithm_file_path = os.path.join(ALGORITHMS_CONFIG_DATA_PATH, ALGORITHM_FILENAME)
    with open(algorithm_file_path, "r") as file:
        lines = file.readlines()
        if any(algorithm_name == line.strip() for line in lines):
            raise Exception("Algorithm already defined in enum.")

    with open(algorithm_file_path, "a") as file:
        file.write(f"    {algorithm_name} = \"{algorithm_name}\"\n")


def modify_functions(algorithm_name, class_name, module_name):
    algorithm_functions_path = os.path.join(ALGORITHMS_CONFIG_DATA_PATH, ALGORITHM_FUNCTIONS_FILENAME)
    with open(algorithm_functions_path, "r") as file:
        content = file.read()
        if algorithm_name in content:
            raise Exception("Algorithm already defined in functions.")
    import_line = f"from bdgs.algorithms.{module_name}.{module_name} import {class_name}"
    map_line = f"ALGORITHM.{algorithm_name}: {class_name}(),"
    with open(algorithm_functions_path, "r") as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if "from bdgs.data.algorithm import ALGORITHM" in line:
            lines.insert(i, import_line + "\n")
            break
    start_index = 0
    for i, line in enumerate(lines):
        if "ALGORITHM_FUNCTIONS = {" in line:
            start_index = i
            break
    for i in range(start_index, len(lines)):
        if lines[i].strip() == "}":
            lines.insert(i, f"    {map_line}\n")
            break
    with open(algorithm_functions_path, "w") as file:
        file.writelines(lines)


def create_algorithm_skeleton(module_name, class_name):
    directory_path = os.path.join(ALGORITHM_BASE_PATH, module_name)
    os.makedirs(directory_path, exist_ok=True)

    file_path = os.path.join(directory_path, f"{module_name}.py")

    skeleton_code = f"""from numpy import ndarray
from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.data.gesture import GESTURE
from bdgs.data.processing_method import PROCESSING_METHOD
from bdgs.models.image_payload import ImagePayload
from bdgs.models.learning_data import LearningData


class {class_name}(BaseAlgorithm):
    def process_image(self, payload: ImagePayload,
                      processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> ndarray:
        raise NotImplementedError("Method process_image not implemented")

    def classify(self, payload: ImagePayload, custom_model_path = None,
                 processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> GESTURE:
        raise NotImplementedError("Method classify not implemented")

    def learn(self, learning_data: list[LearningData], target_model_path: str) -> (float, float):
        raise NotImplementedError("Method learn not implemented")
        
"""
    with open(file_path, "w") as f:
        f.write(skeleton_code)

    with open(os.path.join(directory_path, '__init__.py'), "w") as f:
        f.write('')


def add_custom_payload(module_name, class_name):
    directory_path = os.path.join(ALGORITHM_BASE_PATH, module_name)
    os.makedirs(directory_path, exist_ok=True)

    file_path = os.path.join(directory_path, f"{module_name}_payload.py")

    skeleton_code = f"""import numpy as np

from bdgs.models.image_payload import ImagePayload


class {class_name}Payload(ImagePayload):
    def __init__(self, image: np.ndarray):
        super().__init__(image)
    """

    with open(file_path, "w") as f:
        f.write(skeleton_code)


def add_custom_learning_data(module_name, class_name):
    directory_path = os.path.join(ALGORITHM_BASE_PATH, module_name)
    os.makedirs(directory_path, exist_ok=True)

    file_path = os.path.join(directory_path, f"{module_name}_learning_data.py")

    skeleton_code = f"""from bdgs.data.gesture import GESTURE
from bdgs.models.learning_data import LearningData


class {class_name}LearningData(LearningData):
    def __init__(self, image_path: str, label: GESTURE):
        super().__init__(image_path, label)
    """

    with open(file_path, "w") as f:
        f.write(skeleton_code)


def modify_payload_script(module_name, class_name, algorithm_name):
    payload_import = f"from bdgs.algorithms.{module_name}.{module_name}_payload import {class_name}Payload"
    payload_condition = f"""    elif algorithm == ALGORITHM.{algorithm_name.upper()}:
        payload = {class_name}Payload(image=image)"""

    with open(PAYLOAD_SCRIPTS_FILE, "r") as file:
        lines = file.readlines()

    if payload_import + "\n" not in lines:
        for i, line in enumerate(lines):
            if line.startswith("from bdgs.data.algorithm import ALGORITHM"):
                lines.insert(i, payload_import + "\n")
                break

        for i, line in enumerate(lines):
            if "else:" in line:
                lines.insert(i, payload_condition + "\n")
                break

        with open(PAYLOAD_SCRIPTS_FILE, "w") as file:
            file.writelines(lines)


def modify_learning_data_script(module_name, class_name, algorithm_name):
    learning_import = f"from bdgs.algorithms.{module_name}.{module_name}_learning_data import {class_name}LearningData"
    learning_condition = f"""    elif algorithm == ALGORITHM.{algorithm_name.upper()}:
            return {class_name}LearningData(image_path=image_path, label=label)"""

    with open(LEARNING_DATA_SCRIPTS_FILE, "r") as file:
        lines = file.readlines()

    if learning_import + "\n" not in lines:
        for i, line in enumerate(lines):
            if line.startswith("from bdgs.data.algorithm import ALGORITHM"):
                lines.insert(i, learning_import + "\n")
                break

        for i, line in enumerate(lines):
            if "else:" in line:
                lines.insert(i, learning_condition + "\n")
                break

        with open(LEARNING_DATA_SCRIPTS_FILE, "w") as file:
            file.writelines(lines)


if __name__ == "__main__":
    add_algorithm()
