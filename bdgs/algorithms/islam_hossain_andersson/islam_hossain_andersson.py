import os

import cv2
import keras
import numpy as np

from bdgs.algorithms.bdgs_algorithm import BaseAlgorithm
from bdgs.data.gesture import GESTURE
from bdgs.data.processing_method import PROCESSING_METHOD
from bdgs.algorithms.islam_hossain_andersson.islam_hossain_andersson_payload import IslamHossainAnderssonPayload
from scripts.common.vars import TRAINED_MODELS_PATH


class IslamHossainAndersson(BaseAlgorithm):
    def process_image(self, payload: IslamHossainAnderssonPayload,
                      processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> np.ndarray:
        if processing_method == PROCESSING_METHOD.DEFAULT or processing_method == PROCESSING_METHOD.ISLAM_HOSSAIN_ANDERSSON:
            image = payload.image
            background = payload.bg_image
            # The paper did not specify exact parameters for preprocessing methods, so
            # values used here were selected based on experiments to achieve best results

            # Zoran Zivkovic method to subtract the background. 
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=2, varThreshold=16)
            bg_subtractor.apply(background)
            fg_mask = bg_subtractor.apply(image)
            fg_color = cv2.bitwise_and(image, image, mask=fg_mask)
            # grayscale
            grayscale = cv2.cvtColor(fg_color, cv2.COLOR_BGR2GRAY)
            # morphological erosion
            kernel = np.ones((5,5),np.uint8)
            erosion = cv2.erode(grayscale, kernel)
            # median filter
            median_filter = cv2.medianBlur(erosion, 5)
            #resize
            resized = cv2.resize(median_filter, (50, 50))

            return resized
        else:
            raise Exception("Invalid processing method")

    def classify(self, payload: IslamHossainAnderssonPayload, processing_method: PROCESSING_METHOD = PROCESSING_METHOD.DEFAULT) -> (
            GESTURE, int):
        
        return GESTURE.LIKE, 100
