import cv2
import numpy as np
import time
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QMessageBox
)
from PyQt5.QtCore import QTimer
import sys
from collections import deque

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

# Keypoint mapping
NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER = 0, 1, 2, 3, 4, 5, 6

def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

def incenter(a, b, c):
    side_a = euclidean(b, c)
    side_b = euclidean(a, c)
    side_c = euclidean(a, b)
    px = (side_a * a[0] + side_b * b[0] + side_c * c[0]) / (side_a + side_b + side_c)
    py = (side_a * a[1] + side_b * b[1] + side_c * c[1]) / (side_a + side_b + side_c)
    return np.array([px, py])

def analyze_posture(kp, frame):
    left_ear, right_ear = kp[LEFT_EAR], kp[RIGHT_EAR]
    left_shoulder, right_shoulder = kp[LEFT_SHOULDER], kp[RIGHT_SHOULDER]

    left_face_mid = incenter(kp[LEFT_EYE], kp[LEFT_EAR], kp[NOSE])
    right_face_mid = incenter(kp[RIGHT_EYE], kp[RIGHT_EAR], kp[NOSE])

    s1 = euclidean(left_ear, left_shoulder)
    s2 = euclidean(right_ear, right_shoulder)
    s3 = euclidean(left_shoulder, right_shoulder)
    f2 = (s1 + s2) / s3 if s3 > 0 else 0

    baseline_min, baseline_max, error_margin = 1.2, 1.4, 0.1

    if f2 < (baseline_min - error_margin):
        status, color = "Hunched!", (0, 0, 255)
    elif f2 > (baseline_max + error_margin):
        status, color = "Leaning Back", (0, 165, 255)
    else:
        status, color = "Good Posture", (0, 255, 0)

    cv2.putText(frame, f"f2: {f2:.2f} - {status}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return frame, status

class PostureApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Caretaker 🎀")
        self.resize(250, 120)

        layout = QVBoxLayout()
        self.status_label = QLabel("Monitoring posture...")
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

        self.check_interval = 3  # seconds
        self.bad_posture_log = deque(maxlen=5)
        self.bad_start_time = None
        self.notification_sent = False
        self.snooze_until = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.check_posture)
        self.timer.start(self.check_interval * 1000)

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
            self.status_label.setText("Monitoring posture...")

    def show_snooze_dialog(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Posture Reminder")
        msg.setText("💡 You've been slouching for 30+ seconds.\nPlease sit up straight 🎀")
        snooze_btn = msg.addButton("Remind me later (10 min)", QMessageBox.ActionRole)
        ok_btn = msg.addButton("OK", QMessageBox.AcceptRole)

        msg.exec_()

        if msg.clickedButton() == snooze_btn:
            self.snooze_until = time.time() + 10 * 60  # 10 minutes
            self.notification_sent = True
        elif msg.clickedButton() == ok_btn:
            self.notification_sent = True

    def check_posture(self):
        if not self.running or time.time() < self.snooze_until:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.status_label.setText("Camera error.")
            return

        frame_resized = cv2.resize(frame, (640, 360))
        results = model(frame_resized, verbose=False)

        for r in results:
            if r.keypoints is None:
                continue
            kp = r.keypoints.xy.cpu().numpy()[0]
            frame, status = analyze_posture(kp, frame)

            is_bad = status in ["Hunched!", "Leaning Back"]
            self.bad_posture_log.append((time.time(), is_bad))
            recent_bad = sum(1 for _, bad in self.bad_posture_log if bad)

            if is_bad:
                if self.bad_start_time is None:
                    self.bad_start_time = time.time()
                elapsed = time.time() - self.bad_start_time

                if recent_bad >= 4 and elapsed >= 30 and not self.notification_sent:
                    self.show_snooze_dialog()
            else:
                self.bad_start_time = None
                self.notification_sent = False

        cv2.imshow("YOLOv8 Posture Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            self.close_app()

    def close_app(self):
        self.cap.release()
        cv2.destroyAllWindows()
        QApplication.quit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PostureApp()
    window.show()
    sys.exit(app.exec_())
