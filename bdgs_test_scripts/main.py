import cv2
from bdgs import recognize
from bdgs.classifier import ALGORITHM
from bdgs.gesture import GESTURE

image = cv2.imread("resources/image.jpg")

result = recognize(image, algorithm=ALGORITHM.ALG_1)

print(list(map(str, GESTURE)))
print(result)