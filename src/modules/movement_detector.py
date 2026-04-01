import cv2
import numpy as np

class MovementDetector:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    def detect_movement(self, frame):
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Threshold the mask
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        movements = []

        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter noise
                movements.append(cv2.boundingRect(contour))

        return movements

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)  # Change to video file path if needed
    detector = MovementDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        movements = detector.detect_movement(frame)

        # Draw rectangles around detected movements
        for (x, y, w, h) in movements:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Movement Detection', frame)

        if cv2.waitKey(30) & 0xFF == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()