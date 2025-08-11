import cv2
import mediapipe as mp
import time
from collections import deque
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QMessageBox
from PyQt5.QtCore import QTimer
import sys
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# Constants
IRIS_DIAMETER_MM = 11.7   # average iris diameter in mm
FOCAL_LENGTH_PX = 900     # assumed focal length in pixels (typical laptop webcam)
DISTANCE_THRESHOLD_CM = 60  # bad if closer than this

def estimate_distance(landmarks, w, h):
    """Estimate distance to camera using iris landmarks."""
    # Left and right iris centers
    left_iris = np.array([landmarks[468].x * w, landmarks[468].y * h])
    right_iris = np.array([landmarks[473].x * w, landmarks[473].y * h])

    iris_pixel_dist = np.linalg.norm(left_iris - right_iris)
    if iris_pixel_dist == 0:
        return None

    # d (mm) = f * D / dp
    distance_mm = (FOCAL_LENGTH_PX * IRIS_DIAMETER_MM) / iris_pixel_dist
    return distance_mm / 10  # convert to cm

class EyeDistanceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Caretaker 🎀 Eye Distance Monitor")
        self.resize(300, 120)

        layout = QVBoxLayout()
        self.status_label = QLabel("Monitoring eye distance...")
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)

        layout.addWidget(self.status_label)
        layout.addWidget(self.pause_button)
        self.setLayout(layout)

        self.running = True
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.setText("Error: Cannot access camera.")
            self.running = False

        self.check_interval = 3  # check every 3 seconds
        self.bad_distance_log = deque(maxlen=5)
        self.bad_start_time = None
        self.notification_sent = False
        self.snooze_until = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.check_distance)
        self.timer.start(self.check_interval * 1000)

        self.face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

    def toggle_pause(self):
        if self.running:
            self.running = False
            self.cap.release()
            self.pause_button.setText("Resume")
            self.status_label.setText("Paused.")
        else:
            self.running = True
            self.cap = cv2.VideoCapture(0)
            self.pause_button.setText("Pause")
            self.status_label.setText("Monitoring eye distance...")

    def show_snooze_dialog(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Eye Distance Alert")
        msg.setText("💡 You're sitting too close (<60 cm).\nPlease move back 🎀")
        snooze_btn = msg.addButton("Remind me later (10 min)", QMessageBox.ActionRole)
        ok_btn = msg.addButton("OK", QMessageBox.AcceptRole)

        msg.exec_()
        if msg.clickedButton() == snooze_btn:
            self.snooze_until = time.time() + 10 * 60
            self.notification_sent = True
        elif msg.clickedButton() == ok_btn:
            self.notification_sent = True

    def check_distance(self):
        if not self.running or time.time() < self.snooze_until:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_label.setText("Camera error.")
            return

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                distance_cm = estimate_distance(face_landmarks.landmark, w, h)
                if distance_cm:
                    is_bad = distance_cm < DISTANCE_THRESHOLD_CM
                    self.bad_distance_log.append((time.time(), is_bad))
                    recent_bad = sum(1 for _, bad in self.bad_distance_log if bad)

                    if is_bad:
                        if self.bad_start_time is None:
                            self.bad_start_time = time.time()
                        elapsed = time.time() - self.bad_start_time

                        if recent_bad >= 4 and elapsed >= 30 and not self.notification_sent:
                            self.show_snooze_dialog()
                    else:
                        self.bad_start_time = None
                        self.notification_sent = False

                    status = f"Distance: {distance_cm:.1f} cm"
                    self.status_label.setText(status)
                    cv2.putText(frame, status, (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255) if is_bad else (0, 255, 0), 2)

        cv2.imshow("Eye Distance Monitor", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            self.close_app()

    def close_app(self):
        self.cap.release()
        cv2.destroyAllWindows()
        QApplication.quit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EyeDistanceApp()
    window.show()
    sys.exit(app.exec_())
