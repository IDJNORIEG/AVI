import cv2
import dlib
import numpy as np

class DrowsinessDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.closed_eyes_counter = 0
        self.threshold = 3

    def eye_aspect_ratio(self, eye):
        # Compute the eye aspect ratio
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def detect_drowsiness(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        for face in faces:
            landmarks = self.predictor(gray, face)
            left_eye = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                 (landmarks.part(37).x, landmarks.part(37).y),
                                 (landmarks.part(38).x, landmarks.part(38).y),
                                 (landmarks.part(39).x, landmarks.part(39).y),
                                 (landmarks.part(40).x, landmarks.part(40).y),
                                 (landmarks.part(41).x, landmarks.part(41).y)])
            right_eye = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                  (landmarks.part(43).x, landmarks.part(43).y),
                                  (landmarks.part(44).x, landmarks.part(44).y),
                                  (landmarks.part(45).x, landmarks.part(45).y),
                                  (landmarks.part(46).x, landmarks.part(46).y),
                                  (landmarks.part(47).x, landmarks.part(47).y)])

            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < 0.25:
                self.closed_eyes_counter += 1
                if self.closed_eyes_counter >= self.threshold:
                    cv2.putText(frame, "DROWSY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                self.closed_eyes_counter = 0

        return frame

# Example of how to use the DrowsinessDetector
# detector = DrowsinessDetector()
# while True:
#     ret, frame = video_capture.read()
#     frame = detector.detect_drowsiness(frame)
#     cv2.imshow('Drowsiness Detection', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# video_capture.release()
# cv2.destroyAllWindows()