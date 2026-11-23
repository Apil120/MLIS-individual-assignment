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

# ==========================================
# Example Usage: Run with Webcam
# ==========================================
if __name__ == "__main__":
    print("Initializing detector...")
    detector = create_detector()
    print("Detector ready! Press 'q' to quit.")

    cap = cv2.VideoCapture(0)
    
    # Optional: Set a specific webcam resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Process the frame
        result_frame = detect_hand(frame, detector)

        # Display the result
        cv2.imshow("Hand Detection (300x300 Box)", result_frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()