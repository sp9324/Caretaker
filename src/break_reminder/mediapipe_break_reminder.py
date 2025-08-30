import cv2
import mediapipe as mp
import time
from PyQt5.QtWidgets import QApplication, QMessageBox, QWidget
from PyQt5.QtCore import QTimer
import sys

mp_pose = mp.solutions.pose

class BreakReminder(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Caretaker 🎀 Break Reminder")

        # Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Camera not accessible.")
            sys.exit(1)

        # Timers
        # self.break_interval = 90 * 60  # 90 minutes in seconds
        self.break_interval = 10  
        self.next_break_time = time.time() + self.break_interval
        self.snooze_until = 0
        self.check_interval = 1.5  # seconds between pose checks
        self.check_duration = 60  # seconds to check after reminder
        self.check_end_time = None
        self.standing_detected = False

        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(1000)  # every 1 sec

    def show_reminder(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Break Time")
        msg.setText("💡 Time to stand up and walk for a bit!\nPlease move away from the desk 🎀")
        snooze_btn = msg.addButton("Remind me later (10 min)", QMessageBox.ActionRole)
        ok_btn = msg.addButton("OK", QMessageBox.AcceptRole)
        msg.exec_()

        if msg.clickedButton() == snooze_btn:
            self.snooze_until = time.time() + 10 * 60  # snooze 10 minutes
        else:
            self.start_check()

    def start_check(self):
        self.check_end_time = time.time() + self.check_duration
        self.standing_detected = False

    def detect_standing(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            # Face & chest keypoints: Nose (0), Left Shoulder (11), Right Shoulder (12)
            nose_y = results.pose_landmarks.landmark[0].y * h
            left_shoulder_y = results.pose_landmarks.landmark[11].y * h
            right_shoulder_y = results.pose_landmarks.landmark[12].y * h

            # Heuristic: if both shoulders and nose are out of frame (y < 0 or y > frame height), user likely stood up
            if nose_y < 0 or left_shoulder_y < 0 or right_shoulder_y < 0:
                return True
        else:
            # No landmarks detected → user fully out of frame
            return True

        return False

    def update_loop(self):
        now = time.time()

        # Time for a new break reminder
        if now >= self.next_break_time and now >= self.snooze_until and self.check_end_time is None:
            self.show_reminder()

        # Checking standing after reminder
        if self.check_end_time and now <= self.check_end_time:
            ret, frame = self.cap.read()
            if not ret:
                return

            if self.detect_standing(frame):
                self.standing_detected = True
                print("✅ Standing detected!")
                self.check_end_time = None
                self.next_break_time = time.time() + self.break_interval

        # Check window expired without standing
        elif self.check_end_time and now > self.check_end_time:
            if not self.standing_detected:
                print("⚠ No standing detected. Please take a break!")
            self.check_end_time = None
            self.next_break_time = time.time() + self.break_interval

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BreakReminder()
    window.show()
    sys.exit(app.exec_())
