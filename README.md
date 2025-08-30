<!-- launch -->
1. source venv/bin/activate
2. ./venv/bin/python file_name_daal_do_dhar.py

<!-- push branch every time a new one is created: git push -u origin "branch-name" -->

# Caretaker
An interactive, effective wellbeing app for laptops.

<!-- POSTURE -->
uses: https://www.researchgate.net/publication/383198733_Computer_Vision-Based_Human_Body_Posture_Correction_System/fulltext/66c166c2145f4d3553620b3c/Computer-Vision-Based-Human-Body-Posture-Correction-System.pdf

optimizations
1. uses yolo nano
2. checks posture every 2-3 seconds and not continuously
3. downscaled frame

Preventing false positives:
Keep a timestamped queue of posture checks.
If user is "Hunched!" or "Leaning Back" in ≥4 out of the last 5 checks
AND the bad streak has lasted for ≥30 seconds,
→ trigger notification.
Reset the timer when posture improves.

Caretaker 🎀 – Posture Monitoring App
This is a desktop app that uses your webcam and YOLOv8 pose detection to track your sitting posture in real time.
It will warn you if you sit in a bad posture for too long.

✨ Key Features
Real-time posture tracking using YOLOv8 pose model.

Detects three states:

✅ Good Posture

⚠️ Leaning Back

❌ Hunched

Bad posture detection rule:

If 4 out of the last 5 posture checks are bad and

Bad posture continues for 30 seconds or more → notification is shown.

Snooze option in notification to pause alerts for 10 minutes.

Pause / Resume posture monitoring any time.

Simple PyQt5 interface with live camera feed.

<!-- EYE DISTANCE -->
uses: 
https://pmc.ncbi.nlm.nih.gov/articles/PMC10920617/
https://research.google/blog/mediapipe-iris-real-time-iris-tracking-depth-estimation/

Since distance to the screen is subtle and we need eye landmarks (iris centers), MediaPipe is better — faster and more battery-friendly for this feature.

no user side calibration for easy use of app.
MEDIAPIPE IRIS
Camera Model:

Assumes average interpupillary distance (6.3 cm).

Uses fixed focal length (1000 px) — can be calibrated for accuracy.

Alert logic:

Runs checks every 3 seconds (CHECK_INTERVAL_SECONDS).

Maintains a sliding window of recent checks (deque with BAD_LOG_MAXLEN).

Shows warning if:

Distance < 35 cm (DISTANCE_THRESHOLD_CM).

Condition persists for ≥ 30 seconds.

At least 4 recent measurements are “too close.”

UI Threading: Uses QTimer to schedule distance checks without blocking PyQt’s event loop.
yaw, nose left or right about an axis running up and down; pitch, nose up or down about an axis running from wing to wing

**FUTURE TASKS**
1. tighter accuracy, build a small JSON registry that maps camera model + resolution → f_px.
At app start, read the camera name (e.g., “FaceTime HD Camera (Built-in)”) and resolution, look up f_px. If unknown, fall back to Option A. You (as the developer) do this once per popular laptop model; users don’t calibrate.
