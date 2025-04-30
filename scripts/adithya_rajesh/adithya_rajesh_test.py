from bdgs.data.algorithm import ALGORITHM
from scripts.common.camera_test import camera_test

if __name__ == "__main__":
    # image_processing_test(algorithm=ALGORITHM.ADITHYA_RAJESH)
    # classification_test(algorithm=ALGORITHM.ADITHYA_RAJESH)
    camera_test(algorithm=ALGORITHM.ADITHYA_RAJESH, show_prediction_tresh=70)
