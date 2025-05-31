import logging
import sys

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

from draw import draw_landmarks_on_image

logger = logging.getLogger(__name__)

MODEL_PATH = "data/hand_landmarker.task"
# Global variable to store the detection result
detection_result = None


def process_result(result, output_image: mp.Image, timestamp_ms: int):
    """
    Callback function to process the detection result.
    This function is called in the second thread when the hand landmarks are detected.
    """
    global detection_result
    detection_result = result


def process_images(cap, detector):
    ms = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            logger.info("Ignoring empty camera frame.")
            continue

        mp_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=mp_image)

        ms += 1
        detector.detect_async(mp_image, ms)
        if detection_result:
            logger.info("Hand detector result: %s", detection_result)
            image = draw_landmarks_on_image(image, detection_result, cv2)

        cv2.imshow("Hand Tracking", image)
        if cv2.waitKey(1) & 0xFF == 27:
            break


def main():
    detector_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=MODEL_PATH, delegate=mp.tasks.BaseOptions.Delegate.GPU
        ),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        result_callback=process_result,
        num_hands=2,
    )
    detector = vision.HandLandmarker.create_from_options(detector_options)
    cap = cv2.VideoCapture(0)

    process_images(cap, detector)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main()
