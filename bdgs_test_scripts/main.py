import cv2
from bdgs import recognize
from bdgs.classifier import ALGORITHM
from bdgs.gesture import Gesture

image = cv2.imread("resources/image.jpg")

result = recognize(image, algorithm=ALGORITHM.ALG_1)

print(list(map(str, Gesture)))
print(result)