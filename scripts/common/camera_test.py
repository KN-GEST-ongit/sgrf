import cv2
from tensorflow.python.data.experimental.ops.testing import sleep

from bdgs import classify
from bdgs.algorithms.murthy_jadon.murthy_jadon_payload import MurthyJadonPayload
from bdgs.algorithms.islam_hossain_andersson.islam_hossain_andersson_payload import IslamHossainAnderssonPayload
from bdgs.classifier import process_image
from bdgs.data.algorithm import ALGORITHM
from bdgs.models.image_payload import ImagePayload


def camera_test(algorithm: ALGORITHM, show_prediction_tresh=70):
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
        elif algorithm == ALGORITHM.ISLAM_HOSSAIN_ANDERSSON:
            payload = IslamHossainAnderssonPayload(image=image, bg_image=background)
        # other algs
        else:
            payload = ImagePayload(image=image)

        processed = process_image(algorithm=algorithm, payload=payload)
        prediction, certainty = classify(algorithm=algorithm, payload=payload)

        if certainty >= show_prediction_tresh:
            show_prediction_text(certainty, image, prediction)
        show_processed(image, processed)
        cv2.imshow("Classifier", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def show_processed(image, processed):
    if len(processed.shape) == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    processed = processed.squeeze()
    if processed.shape[0] == 0 or processed.shape[1] == 0:
        print("Processed image has zero dimensions. Skipping thumbnail generation.")
    else:
        thumbnail_height = 150
        thumbnail_width = int(processed.shape[1] * (thumbnail_height / processed.shape[0]))
        thumbnail = cv2.resize(processed, (thumbnail_width, thumbnail_height))

        if len(thumbnail.shape) == 2:
            thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_GRAY2BGR)

        image[0:thumbnail.shape[0], 0:thumbnail.shape[1]] = thumbnail


def show_prediction_text(certainty, image, prediction):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"{str(prediction)} ({certainty}%)"
    font_scale = 0.8
    thickness = 2
    color = (255, 255, 255)
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = image.shape[1] - text_width - 10
    y = image.shape[0] - 10
    cv2.putText(image, text, (x, y), font, font_scale, color, thickness)
