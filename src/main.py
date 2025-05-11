import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

from draw import draw_landmarks_on_image

MODEL_PATH = "data/hand_landmarker.task"


def process_images(cap, detector):
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        mp_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=mp_image)
        detection_result = detector.detect(mp_image)

        annotated_image = draw_landmarks_on_image(image, detection_result, cv2)

        cv2.imshow("Hand Tracking", annotated_image)
        if cv2.waitKey(1) & 0xFF == 27:
            break


def main():
    detector_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=MODEL_PATH, delegate=mp.tasks.BaseOptions.Delegate.GPU
        ),
        num_hands=2,
    )
    detector = vision.HandLandmarker.create_from_options(detector_options)
    cap = cv2.VideoCapture(0)

    process_images(cap, detector)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
