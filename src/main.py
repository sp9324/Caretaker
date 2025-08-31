import sys
import threading
from PyQt5.QtWidgets import QApplication
from eye_distance.mediapipe_eye_distance import EyeDistanceApp
from posture.yolo_posture import PostureMonitor

def run_posture_monitor():
    posture = PostureMonitor()
    posture.start()

if __name__ == "__main__":
    # Start posture monitor in background thread
    posture_thread = threading.Thread(target=run_posture_monitor, daemon=True)
    posture_thread.start()

    # Start Eye Distance GUI
    app = QApplication(sys.argv)
    window = EyeDistanceApp()
    window.show()
    sys.exit(app.exec_())
