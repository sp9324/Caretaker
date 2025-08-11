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

no calibration for easy use of app.
MEDIAPIPE IRIS
Uses MediaPipe FaceMesh for iris landmarks (468 and 473).
Iris diameter (D) = 11.7 mm (standard from MediaPipe research).
Focal length (f) = assume ~900 pixels (reasonable for typical 720p/1080p laptop webcams).
Bad condition = distance < 60 cm.
Notification trigger = if 4 out of last 5 checks are bad AND bad streak lasts ≥ 30 seconds.
Includes pause/resume and snooze for 10 minutes like in your posture code.
