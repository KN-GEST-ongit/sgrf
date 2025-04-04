import cv2
from tensorflow.python.data.experimental.ops.testing import sleep

from bdgs import classify
from bdgs.algorithms.murthy_jadon.murthy_jadon_payload import MurthyJadonPayload
from bdgs.classifier import ALGORITHM
from bdgs.models.image_payload import ImagePayload


def camera_test(algorithm: ALGORITHM):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return
    sleep(500)

    ret, background = cap.read()
    if not ret:
        print("Cannot read camera.")
        cap.release()
        return

    image = background.copy()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 2 == 0:
            image = frame

        if algorithm == ALGORITHM.MURTHY_JADON:
            payload = MurthyJadonPayload(image=image, bg_image=background)
        # other
        else:
            payload = ImagePayload(image=image)

        prediction = classify(algorithm=algorithm, payload=payload)

        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(image, str(prediction), (0, 100), font, 1, (255, 255, 255), 2)
        cv2.imshow("Image", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
