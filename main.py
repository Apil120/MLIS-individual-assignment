## Main entry point - MediaPipe Hand Detection + NSL Character Classification

import cv2
import mediapipe as mp
from utils import create_detector, draw_hand_landmarks
from classifier import NSLClassifier


def capture_camera_feed():
    """Capture camera feed, detect hand, and classify NSL character."""
    # Initialize MediaPipe detector
    detector = create_detector()

    # Initialize NSL Classifier
    classifier = NSLClassifier(model_path="notebooks/best_model.pth")

    cap = cv2.VideoCapture(0)
    print("Starting camera feed... Press 'q' to quit")
    print("Show an NSL sign to the camera for classification")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Detect hand landmarks using MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        annotated_frame = frame.copy()
        prediction_text = "No hand detected"
        prediction_color = (0, 0, 255)  # Red

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Draw landmarks and connections
                draw_hand_landmarks(annotated_frame, hand_landmarks)

                # Get hand crop and classify
                crop, bbox = classifier.get_hand_crop(frame, hand_landmarks)

                if crop.size > 0:
                    class_idx, char, confidence, _ = classifier.predict(crop)

                    # Format prediction text
                    prediction_text = f"{char} (Class {class_idx}) - {confidence:.2%}"
                    prediction_color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)

                    # Draw bounding box
                    x_min, y_min, x_max, y_max = bbox
                    cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

        # Draw prediction on frame
        cv2.putText(annotated_frame, prediction_text, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, prediction_color, 2)

        # Draw instructions
        cv2.putText(annotated_frame, "Press 'q' to quit", (20, annotated_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('NSL Recognition - MediaPipe + CNN', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_camera_feed()