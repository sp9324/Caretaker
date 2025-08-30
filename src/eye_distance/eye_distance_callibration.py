# import cv2
# import mediapipe as mp
# import numpy as np

# mp_face_mesh = mp.solutions.face_mesh
# mp_drawing = mp.solutions.drawing_utils

# # FaceMesh configuration
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
#                                   max_num_faces=1,
#                                   refine_landmarks=True,
#                                   min_detection_confidence=0.5,
#                                   min_tracking_confidence=0.5)

# # 3D model points for head pose estimation (approx landmarks)
# # Nose tip, chin, left eye corner, right eye corner, left mouth corner, right mouth corner
# FACE_3D_POINTS = np.array([
#     (0.0, 0.0, 0.0),        # Nose tip
#     (0.0, -63.6, -12.5),    # Chin
#     (-43.3, 32.7, -26.0),   # Left eye left corner
#     (43.3, 32.7, -26.0),    # Right eye right corner
#     (-28.9, -28.9, -24.1),  # Left Mouth corner
#     (28.9, -28.9, -24.1)    # Right mouth corner
# ], dtype=np.float64)

# def get_head_pose_and_corrected_eye_distance(image, landmarks, w, h):
#     # 2D landmarks (from mediapipe, mapped to image coords)
#     FACE_2D_POINTS = np.array([
#         (landmarks[1].x * w, landmarks[1].y * h),     # Nose tip
#         (landmarks[152].x * w, landmarks[152].y * h), # Chin
#         (landmarks[263].x * w, landmarks[263].y * h), # Right eye right corner
#         (landmarks[33].x * w, landmarks[33].y * h),   # Left eye left corner
#         (landmarks[287].x * w, landmarks[287].y * h), # Right Mouth corner
#         (landmarks[57].x * w, landmarks[57].y * h),   # Left Mouth corner
#     ], dtype=np.float64)

#     # Camera matrix (assuming no lens distortion)
#     focal_length = w
#     cam_matrix = np.array([[focal_length, 0, w / 2],
#                            [0, focal_length, h / 2],
#                            [0, 0, 1]])

#     dist_matrix = np.zeros((4, 1), dtype=np.float64)

#     # SolvePnP to get rotation + translation vectors
#     success, rot_vec, trans_vec = cv2.solvePnP(FACE_3D_POINTS, FACE_2D_POINTS,
#                                                cam_matrix, dist_matrix,
#                                                flags=cv2.SOLVEPNP_ITERATIVE)

#     # Project eyes into corrected 3D space
#     left_eye_idx = [33, 133]   # left eye corner landmarks
#     right_eye_idx = [362, 263] # right eye corner landmarks

#     left_eye = np.mean([(landmarks[i].x * w, landmarks[i].y * h) for i in left_eye_idx], axis=0)
#     right_eye = np.mean([(landmarks[i].x * w, landmarks[i].y * h) for i in right_eye_idx], axis=0)

#     # Convert 2D → 3D using projection
#     eye_2d = np.array([left_eye, right_eye], dtype=np.float64).reshape(-1, 1, 2)
#     eye_3d, _ = cv2.projectPoints(np.array([[0,0,0]], dtype=np.float64),
#                                   rot_vec, trans_vec, cam_matrix, dist_matrix)

#     # Estimate depth correction by inverse projection
#     rmat, _ = cv2.Rodrigues(rot_vec)
#     proj_matrix = np.hstack((rmat, trans_vec))
#     _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

#     # Roll, pitch, yaw correction
#     pitch, yaw, roll = [np.radians(angle) for angle in euler_angles]

#     # Approximate corrected Euclidean eye distance
#     eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
#     corrected_distance = eye_distance / (np.cos(yaw) * np.cos(pitch))

#     return corrected_distance, (left_eye, right_eye)

# def main():
#     cap = cv2.VideoCapture(0)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         h, w, _ = frame.shape
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(rgb_frame)

#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 # Get corrected eye distance
#                 distance, (left_eye, right_eye) = get_head_pose_and_corrected_eye_distance(frame, face_landmarks.landmark, w, h)

#                 # Draw eyes
#                 cv2.circle(frame, tuple(np.int32(left_eye)), 3, (0,255,0), -1)
#                 cv2.circle(frame, tuple(np.int32(right_eye)), 3, (0,255,0), -1)

#                 # Show distance
#                 cv2.putText(frame, f"Corrected Eye Distance: {float(distance):.2f}px", (30, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

#         cv2.imshow("Eye Distance with Head Pose Correction", frame)
#         if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

import cv2
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Focal length and real eye distance assumption (in cm)
FOCAL_LENGTH = 1000  # Approximate focal length (you may need to calibrate)
REAL_EYE_DISTANCE_CM = 6.3  # Average interpupillary distance in cm

def eye_distance(landmarks, img_w, img_h):
    # Left eye landmark (468), Right eye landmark (473)
    left_eye = (int(landmarks[468].x * img_w), int(landmarks[468].y * img_h))
    right_eye = (int(landmarks[473].x * img_w), int(landmarks[473].y * img_h))

    # Pixel distance between eyes
    pixel_dist = ((left_eye[0] - right_eye[0])**2 + (left_eye[1] - right_eye[1])**2) ** 0.5

    # Convert pixel distance to real-world distance (cm) using pinhole camera model
    if pixel_dist != 0:
        distance_cm = (REAL_EYE_DISTANCE_CM * FOCAL_LENGTH) / pixel_dist
    else:
        distance_cm = 0

    return pixel_dist, distance_cm, left_eye, right_eye

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            pixel_dist, distance_cm, left_eye, right_eye = eye_distance(face_landmarks.landmark, w, h)

            # Draw eye points
            cv2.circle(frame, left_eye, 3, (0, 255, 0), -1)
            cv2.circle(frame, right_eye, 3, (0, 255, 0), -1)

            # Draw line between eyes
            cv2.line(frame, left_eye, right_eye, (255, 0, 0), 2)

            # Show distance on screen
            cv2.putText(frame, f"Dist: {distance_cm:.2f} cm", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Debug print
            print(f"Pixel Distance: {pixel_dist:.2f}, Distance (cm): {distance_cm:.2f}")

    cv2.imshow("Eye Distance Measurement", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
