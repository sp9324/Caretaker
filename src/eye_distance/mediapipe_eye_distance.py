# import cv2
# import mediapipe as mp
# import time
# from collections import deque
# from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QMessageBox
# from PyQt5.QtCore import QTimer
# import sys
# import numpy as np
# import math

# mp_face_mesh = mp.solutions.face_mesh

# # Initialize Mediapipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# # Focal length and real eye distance assumption (in cm)
# FOCAL_LENGTH = 1000  # Approximate focal length (you may need to calibrate)
# REAL_EYE_DISTANCE_CM = 6.3  # Average interpupillary distance in cm
# DISTANCE_THRESHOLD_CM=35

# def eye_distance(landmarks, img_w, img_h):
#     # Left eye landmark (468), Right eye landmark (473)
#     left_eye = (int(landmarks[468].x * img_w), int(landmarks[468].y * img_h))
#     right_eye = (int(landmarks[473].x * img_w), int(landmarks[473].y * img_h))

#     # Pixel distance between eyes
#     pixel_dist = ((left_eye[0] - right_eye[0])**2 + (left_eye[1] - right_eye[1])**2) ** 0.5

#     # Convert pixel distance to real-world distance (cm) using pinhole camera model
#     if pixel_dist != 0:
#         distance_cm = (REAL_EYE_DISTANCE_CM * FOCAL_LENGTH) / pixel_dist
#     else:
#         distance_cm = 0

#     return pixel_dist, distance_cm, left_eye, right_eye

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_frame)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             h, w, _ = frame.shape
#             pixel_dist, distance_cm, left_eye, right_eye = eye_distance(face_landmarks.landmark, w, h)

#             # Draw eye points
#             cv2.circle(frame, left_eye, 3, (0, 255, 0), -1)
#             cv2.circle(frame, right_eye, 3, (0, 255, 0), -1)

#             # Draw line between eyes
#             cv2.line(frame, left_eye, right_eye, (255, 0, 0), 2)

#             # Show distance on screen
#             cv2.putText(frame, f"Dist: {distance_cm:.2f} cm", (50, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

#             # Debug print
#             print(f"Pixel Distance: {pixel_dist:.2f}, Distance (cm): {distance_cm:.2f}")

#     cv2.imshow("Eye Distance Measurement", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# class EyeDistanceApp(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Caretaker 🎀 Eye Distance Monitor")
#         self.resize(300, 120)

#         layout = QVBoxLayout()
#         self.status_label = QLabel("Monitoring eye distance...")
#         self.pause_button = QPushButton("Pause")
#         self.pause_button.clicked.connect(self.toggle_pause)

#         layout.addWidget(self.status_label)
#         layout.addWidget(self.pause_button)
#         self.setLayout(layout)

#         self.running = True
#         self.cap = cv2.VideoCapture(0)
#         if not self.cap.isOpened():
#             self.status_label.setText("Error: Cannot access camera.")
#             self.running = False

#         self.check_interval = 3  # check every 3 seconds
#         self.bad_distance_log = deque(maxlen=5)
#         self.bad_start_time = None
#         self.notification_sent = False
#         self.snooze_until = 0

#         self.timer = QTimer()
#         self.timer.timeout.connect(self.check_distance)
#         self.timer.start(self.check_interval * 1000)

#         self.face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

#     def toggle_pause(self):
#         if self.running:
#             self.running = False
#             self.cap.release()
#             self.pause_button.setText("Resume")
#             self.status_label.setText("Paused.")
#         else:
#             self.running = True
#             self.cap = cv2.VideoCapture(0)
#             self.pause_button.setText("Pause")
#             self.status_label.setText("Monitoring eye distance...")

#     def show_snooze_dialog(self):
#         msg = QMessageBox()
#         msg.setIcon(QMessageBox.Warning)
#         msg.setWindowTitle("Eye Distance Alert")
#         msg.setText("💡 You're sitting too close (<60 cm).\nPlease move back 🎀")
#         snooze_btn = msg.addButton("Remind me later (10 min)", QMessageBox.ActionRole)
#         ok_btn = msg.addButton("OK", QMessageBox.AcceptRole)

#         msg.exec_()
#         if msg.clickedButton() == snooze_btn:
#             self.snooze_until = time.time() + 10 * 60
#             self.notification_sent = True
#         elif msg.clickedButton() == ok_btn:
#             self.notification_sent = True

#     def check_distance(self):
#         if not self.running or time.time() < self.snooze_until:
#             return

#         ret, frame = self.cap.read()
#         if not ret:
#             self.status_label.setText("Camera error.")
#             return

#         h, w, _ = frame.shape
#         f_px = FOCAL_LENGTH(w)
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = self.face_mesh.process(rgb_frame)

#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 # distance_cm = estimate_distance(face_landmarks.landmark, w, h)
#                 distance_cm = eye_distance(face_landmarks.landmark, w, h, f_px=f_px)
#                 if distance_cm:
#                     is_bad = distance_cm < DISTANCE_THRESHOLD_CM
#                     self.bad_distance_log.append((time.time(), is_bad))
#                     recent_bad = sum(1 for _, bad in self.bad_distance_log if bad)

#                     if is_bad:
#                         if self.bad_start_time is None:
#                             self.bad_start_time = time.time()
#                         elapsed = time.time() - self.bad_start_time

#                         if recent_bad >= 4 and elapsed >= 30 and not self.notification_sent:
#                             self.show_snooze_dialog()
#                     else:
#                         self.bad_start_time = None
#                         self.notification_sent = False

#                     status = f"Distance: {distance_cm:.1f} cm"
#                     self.status_label.setText(status)
#                     cv2.putText(frame, status, (30, 50),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1,
#                                 (0, 0, 255) if is_bad else (0, 255, 0), 2)

#         cv2.imshow("Eye Distance Monitor", frame)
#         if cv2.waitKey(1) & 0xFF == 27:
#             self.close_app()

#     def close_app(self):
#         self.cap.release()
#         cv2.destroyAllWindows()
#         QApplication.quit()

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = EyeDistanceApp()
#     window.show()
#     sys.exit(app.exec_())

"""
Eye Distance Monitor
Cleaned and refactored version of the user's script.

Requirements:
 - mediapipe
 - opencv-python
 - PyQt5
 - numpy

Run:
 python eye_distance_monitor.py
"""

import sys
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QMessageBox

# ---------------------------
# Configuration / constants
# ---------------------------
FOCAL_LENGTH_PX = 1000         # approximate focal length in pixels (tune/calibrate if needed)
REAL_EYE_DISTANCE_CM = 6.3     # average interpupillary distance (cm)
DISTANCE_THRESHOLD_CM = 35.0   # threshold to consider "too close"
CHECK_INTERVAL_SECONDS = 3     # how often to sample camera (seconds)
BAD_LOG_MAXLEN = 10            # how many recent checks to keep
SNOOZE_SECONDS = 10 * 60       # 10 minutes snooze


# ---------------------------
# Helper functions
# ---------------------------
def landmarks_to_point(landmark, img_w, img_h):
    """Convert a Mediapipe landmark to integer (x, y)."""
    return int(landmark.x * img_w), int(landmark.y * img_h)


def eye_center_from_iris(landmarks, img_w, img_h, left=True):
    """
    Estimate the eye center using Mediapipe iris landmarks.

    For refine_landmarks=True:
      left iris landmarks indices: 468, 469, 470, 471
      right iris landmarks indices: 473, 474, 475, 476

    Returns (x, y) tuple (integers).
    If landmarks are missing or invalid, returns None.
    """
    try:
        if left:
            idxs = [468, 469, 470, 471]
        else:
            idxs = [473, 474, 475, 476]

        pts = [landmarks[i] for i in idxs]
        xs = [(p.x * img_w) for p in pts]
        ys = [(p.y * img_h) for p in pts]
        cx = int(sum(xs) / len(xs))
        cy = int(sum(ys) / len(ys))
        return cx, cy
    except Exception:
        return None


def estimate_distance_cm(left_pt, right_pt, focal_length_px=FOCAL_LENGTH_PX,
                         real_eye_distance_cm=REAL_EYE_DISTANCE_CM):
    """
    Given the left and right eye pixel coordinates, estimate real-world distance (cm)
    using the pinhole camera model:
      distance_cm = (real_eye_distance_cm * focal_length_px) / pixel_distance
    Returns (pixel_distance, distance_cm)
    """
    (lx, ly) = left_pt
    (rx, ry) = right_pt
    pixel_dist = float(np.hypot(lx - rx, ly - ry))
    if pixel_dist <= 0.0:
        return pixel_dist, 0.0
    distance_cm = (real_eye_distance_cm * focal_length_px) / pixel_dist
    return pixel_dist, distance_cm


# ---------------------------
# Main application class
# ---------------------------
class EyeDistanceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Caretaker 🎀 Eye Distance Monitor")
        self.resize(360, 140)

        # UI
        layout = QVBoxLayout()
        self.status_label = QLabel("Initializing...")
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        layout.addWidget(self.status_label)
        layout.addWidget(self.pause_button)
        self.setLayout(layout)

        # State
        self.running = True
        self.snooze_until = 0
        self.notification_sent = False
        self.bad_distance_log = deque(maxlen=BAD_LOG_MAXLEN)
        self.bad_start_time = None

        # Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.setText("Error: Cannot access camera.")
            self.running = False

        # Mediapipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_distance)
        self.timer.start(int(CHECK_INTERVAL_SECONDS * 1000))

        self.status_label.setText("Monitoring eye distance...")

    def toggle_pause(self):
        if self.running:
            self.running = False
            # release camera to free resource
            if self.cap and self.cap.isOpened():
                self.cap.release()
            self.pause_button.setText("Resume")
            self.status_label.setText("Paused.")
        else:
            # re-open camera
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.status_label.setText("Error: Cannot access camera.")
                self.running = False
                return
            self.running = True
            self.pause_button.setText("Pause")
            self.status_label.setText("Monitoring eye distance...")

    def show_snooze_dialog(self, distance_cm):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Eye Distance Alert")
        msg.setText(f"💡 You're sitting too close ({distance_cm:.1f} cm).\nPlease move back 🎀")
        snooze_btn = msg.addButton("Remind me later (10 min)", QMessageBox.ActionRole)
        ok_btn = msg.addButton("OK", QMessageBox.AcceptRole)

        msg.exec_()
        if msg.clickedButton() == snooze_btn:
            self.snooze_until = time.time() + SNOOZE_SECONDS
            self.notification_sent = True
        elif msg.clickedButton() == ok_btn:
            self.notification_sent = True

    def check_distance(self):
        """
        Periodically called by QTimer. Captures a frame, runs Mediapipe, estimates distance,
        updates UI, shows an alert if needed.
        """
        if not self.running or time.time() < self.snooze_until:
            return

        if not (self.cap and self.cap.isOpened()):
            self.status_label.setText("Camera not available.")
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.status_label.setText("Camera read error.")
            return

        img_h, img_w = frame.shape[:2]
        # Convert to RGB for mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        distance_cm = None
        pixel_dist = 0.0
        left_pt = right_pt = None

        if results.multi_face_landmarks:
            # Use first detected face
            face_landmarks = results.multi_face_landmarks[0].landmark

            left_pt = eye_center_from_iris(face_landmarks, img_w, img_h, left=True)
            right_pt = eye_center_from_iris(face_landmarks, img_w, img_h, left=False)

            if left_pt is not None and right_pt is not None:
                pixel_dist, distance_cm = estimate_distance_cm(left_pt, right_pt)
                # draw visualization on frame
                cv2.circle(frame, left_pt, 3, (0, 255, 0), -1)
                cv2.circle(frame, right_pt, 3, (0, 255, 0), -1)
                cv2.line(frame, left_pt, right_pt, (255, 0, 0), 2)
                label = f"Dist: {distance_cm:.1f} cm"
                is_bad = distance_cm < DISTANCE_THRESHOLD_CM
                cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 0, 255) if is_bad else (0, 255, 0), 2)

                # logging for alert decision
                self.bad_distance_log.append((time.time(), bool(is_bad)))
                recent_bad = sum(1 for _, b in self.bad_distance_log if b)

                if is_bad:
                    if self.bad_start_time is None:
                        self.bad_start_time = time.time()
                    elapsed = time.time() - (self.bad_start_time or time.time())
                    # require several recent bad samples and a minimum duration
                    if recent_bad >= 4 and elapsed >= 30 and not self.notification_sent:
                        self.show_snooze_dialog(distance_cm)
                else:
                    # reset
                    self.bad_start_time = None
                    self.notification_sent = False

                # update status label
                self.status_label.setText(f"Distance: {distance_cm:.1f} cm")
            else:
                self.status_label.setText("Face detected, could not find iris landmarks.")
        else:
            self.status_label.setText("No face detected.")
            # reset bad-start if face not present
            self.bad_start_time = None
            self.notification_sent = False

        # Show preview window (optional)
        cv2.imshow("Eye Distance Monitor (press ESC to close)", frame)
        # Allow OpenCV window to process events
        key = cv2.waitKey(1)
        if key & 0xFF == 27:  # ESC pressed
            self.close_app()

    def close_app(self):
        self.timer.stop()
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        # release mediapipe resources
        try:
            self.face_mesh.close()
        except Exception:
            pass
        QApplication.quit()


# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EyeDistanceApp()
    window.show()
    sys.exit(app.exec_())
