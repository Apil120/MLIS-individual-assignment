import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
from pathlib import Path

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path("models/hand_landmarker.task")

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


def draw_hand_landmarks(frame, hand_landmarks):
    """Draw hand landmarks and connections on the frame."""
    h, w, _ = frame.shape

    # Draw landmarks
    for landmark in hand_landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    # Draw connections
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        start = hand_landmarks[start_idx]
        end = hand_landmarks[end_idx]
        start_pt = (int(start.x * w), int(start.y * h))
        end_pt = (int(end.x * w), int(end.y * h))
        cv2.line(frame, start_pt, end_pt, (255, 0, 0), 2)

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
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4
    )
    return vision.HandLandmarker.create_from_options(options)

def draw_hand_rectangle(frame, hand_landmarks):
    """Draw a fixed 300x300 bounding box centered on the hand."""
    h, w, _ = frame.shape
    box_size = 300
    half_size = box_size // 2

    # 1. Calculate the center (centroid) of the hand from all landmarks
    x_coords = [lm.x for lm in hand_landmarks]
    y_coords = [lm.y for lm in hand_landmarks]
    
    center_x_norm = sum(x_coords) / len(x_coords)
    center_y_norm = sum(y_coords) / len(y_coords)
    
    # Convert normalized coordinates (0.0 to 1.0) to pixel coordinates
    center_x_px = int(center_x_norm * w)
    center_y_px = int(center_y_norm * h)

    # 2. Calculate the top-left and bottom-right corners of the 300x300 box
    x_min = center_x_px - half_size
    x_max = center_x_px + half_size
    y_min = center_y_px - half_size
    y_max = center_y_px + half_size

    # 3. Clamp coordinates to ensure the box doesn't draw outside the frame boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    # 4. Draw bounding box and label
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
    cv2.putText(frame, "Hand", (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

def detect_hand(frame, detector):
    """Detect hand and draw the 300x300 bounding box on the frame."""
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

