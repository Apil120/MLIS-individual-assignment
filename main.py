## Main entry point for the application - MediaPipe Hand Detection

import cv2
from utils import create_detector, detect_hand


def capture_camera_feed():
    """Capture camera feed and run hand detection."""
    detector = create_detector()

    cap = cv2.VideoCapture(0)
    print("Starting camera feed... Press 'q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Pass frame to MediaPipe hand detector
        frame = detect_hand(frame, detector)

        cv2.imshow('Video Feed - MediaPipe Hand Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_camera_feed()