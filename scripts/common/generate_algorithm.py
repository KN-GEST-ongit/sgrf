import os.path

from scripts.common.vars import ALGORITHMS_CONFIG_DATA_PATH

ALGORITHM_FILENAME = "algorithm.py"
ALGORITHM_FUNCTIONS_FILENAME = "algorithm_functions.py"
PROCESSING_METHOD_FILENAME = "processing_method.py"


def add_algorithm():
    while True:
        algorithm_name = input("Algorithm name: ").strip().upper()
        if not algorithm_name:
            print("Invalid name.")
            continue

        with open(os.path.join(ALGORITHMS_CONFIG_DATA_PATH, ALGORITHM_FILENAME), "r") as file:
            lines = file.readlines()
            if any(algorithm_name in line for line in lines):
                print("Algorithm already exists.")
                continue

        with open(os.path.join(ALGORITHMS_CONFIG_DATA_PATH, ALGORITHM_FILENAME), "a") as file:
            file.write(f"    {algorithm_name} = \"{algorithm_name}\"\n")

        print(f"Algorithm added: {algorithm_name}")
        return


if __name__ == "__main__":
    add_algorithm()
