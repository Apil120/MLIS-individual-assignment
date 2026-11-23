import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
from pathlib import Path

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


def create_detector():
    """Create and return a MediaPipe HandLandmarker detector."""
    download_model_if_needed()

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return vision.HandLandmarker.create_from_options(options)


def draw_hand_rectangle(frame, hand_landmarks):
    """Draw only a bounding box rectangle around the hand."""
    h, w, _ = frame.shape

    # Calculate bounding box coordinates from landmarks
    x_coords = [lm.x for lm in hand_landmarks]
    y_coords = [lm.y for lm in hand_landmarks]
    
    x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
    y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
    
    # Draw bounding box around hand
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

    # Add label (optional: remove these 2 lines if you want strictly ONLY the rectangle)
    cv2.putText(frame, "Hand", (x_min, y_min - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def detect_hand(frame, detector):
    """Detect hand and draw only the bounding box rectangle on the frame."""
    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect hand landmarks
    detection_result = detector.detect(mp_image)

    # Draw rectangle on the original frame
    annotated_frame = frame.copy()

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            draw_hand_rectangle(annotated_frame, hand_landmarks)

    return annotated_frame