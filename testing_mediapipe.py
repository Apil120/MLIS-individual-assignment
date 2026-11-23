## Main entry point for the application - MediaPipe Hand Detection

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
from pathlib import Path

# Hand landmark connections (21 landmarks, 0-20)
# MediaPipe hand connections: (start, end) pairs
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm connections
]

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path("models/hand_landmarker.task")

def download_model_if_needed():
    """Download MediaPipe hand landmarker model if it doesn't exist."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print(f"Downloading MediaPipe hand landmarker model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete!")
    else:
        print(f"Model already exists at {MODEL_PATH}")

def detect_hand(frame, detector):
    """Detect hand landmarks and draw them on the frame."""
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect hand landmarks
    detection_result = detector.detect(mp_image)

    # Draw landmarks on the original frame
    annotated_frame = frame.copy()

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Draw hand landmarks
            for landmark in hand_landmarks:
                h, w, _ = annotated_frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)

            # Draw connections between landmarks
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                h, w, _ = annotated_frame.shape
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                cv2.line(annotated_frame, start_point, end_point, (255, 0, 0), 2)

            # Draw bounding box around hand
            x_coords = [lm.x for lm in hand_landmarks]
            y_coords = [lm.y for lm in hand_landmarks]
            h, w, _ = annotated_frame.shape
            x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
            y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

            # Add label
            cv2.putText(annotated_frame, "Hand", (x_min, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return annotated_frame


def capture_camera_feed():
    """Capture camera feed and run hand detection."""
    # Download model if needed
    download_model_if_needed()

    cap = cv2.VideoCapture(0)

    # Initialize MediaPipe Hand Landmarker
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)

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