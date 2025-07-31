import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 pose model (nano for speed)
model = YOLO("yolov8n-pose.pt")

# Keypoint mapping (based on your schema)
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6

def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

def incenter(a, b, c):
    """Compute incenter of a triangle given points a, b, c."""
    side_a = euclidean(b, c)
    side_b = euclidean(a, c)
    side_c = euclidean(a, b)
    px = (side_a * a[0] + side_b * b[0] + side_c * c[0]) / (side_a + side_b + side_c)
    py = (side_a * a[1] + side_b * b[1] + side_c * c[1]) / (side_a + side_b + side_c)
    return np.array([px, py])

def analyze_posture(kp, frame):
    """Analyze posture using f2 metric and incenter midpoints."""
    left_ear = kp[LEFT_EAR]
    right_ear = kp[RIGHT_EAR]
    left_shoulder = kp[LEFT_SHOULDER]
    right_shoulder = kp[RIGHT_SHOULDER]

    # Left face incenter: triangle (left_eye, left_ear, nose)
    left_face_mid = incenter(kp[LEFT_EYE], kp[LEFT_EAR], kp[NOSE])
    right_face_mid = incenter(kp[RIGHT_EYE], kp[RIGHT_EAR], kp[NOSE])

    # Distances
    s1 = euclidean(left_ear, left_shoulder)
    s2 = euclidean(right_ear, right_shoulder)
    s3 = euclidean(left_shoulder, right_shoulder)

    f2 = (s1 + s2) / s3 if s3 > 0 else 0

    baseline_min, baseline_max = 1.2, 1.4
    error_margin = 0.1

    if f2 < (baseline_min - error_margin):
        status = "Hunched!"
        color = (0, 0, 255)
    elif f2 > (baseline_max + error_margin):
        status = "Leaning Back"
        color = (0, 165, 255)
    else:
        status = "Good Posture"
        color = (0, 255, 0)

    # Draw keypoints and midpoints
    h, w, _ = frame.shape
    for point in [left_ear, right_ear, left_shoulder, right_shoulder, left_face_mid, right_face_mid]:
        cx, cy = int(point[0]), int(point[1])
        cv2.circle(frame, (cx, cy), 5, color, -1)

    # Put text
    cv2.putText(frame, f"f2: {f2:.2f} - {status}",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return frame, status, f2

def run():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        for r in results:
            if r.keypoints is None:
                continue
            kp = r.keypoints.xy.cpu().numpy()[0]  # first person
            frame, status, f2 = analyze_posture(kp, frame)

        cv2.imshow("YOLOv8 Posture Detection", frame)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
