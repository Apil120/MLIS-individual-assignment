## Main entry point for the application

import cv2
from ultralytics import YOLO

def detect_hand(frame, model):
    results = model.predict(frame, conf=0.5, verbose=False)

    # Draw predictions on the frame
    annotated_frame = results[0].plot()

    return annotated_frame

def capture_camera_feed():
    cap = cv2.VideoCapture(0)

    # Initialize model once, outside the loop
    model = YOLO('models/hand_detector/best.pt')

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Pass frame to YOLO
        frame = detect_hand(frame, model)

        cv2.imshow('Video Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


capture_camera_feed()