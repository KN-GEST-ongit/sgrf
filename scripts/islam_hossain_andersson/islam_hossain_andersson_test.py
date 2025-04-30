from bdgs.data.algorithm import ALGORITHM
from scripts.common.classification_test import classification_test
from scripts.common.image_processing_test import image_processing_test
from scripts.common.camera_test import camera_test


if __name__ == "__main__":
    #image_processing_test(algorithm=ALGORITHM.ISLAM_HOSSAIN_ANDERSSON)
    #classification_test(algorithm=ALGORITHM.ISLAM_HOSSAIN_ANDERSSON)
    camera_test(algorithm=ALGORITHM.ISLAM_HOSSAIN_ANDERSSON, show_prediction_tresh=70)
