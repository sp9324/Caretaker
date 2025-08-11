<!-- launch -->
source venv/bin/activate
python src/posture/yolo_posture.py

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
