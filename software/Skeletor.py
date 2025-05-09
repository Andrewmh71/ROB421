import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose_detection = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    enable_segmentation=False,
    model_complexity=1
)

# Open video (0 for webcam, or path to video file)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Failed to open video source.")

# Read one frame to get dimensions
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read from video source.")

height, width, _ = frame.shape
diag_pixels = np.sqrt(width ** 2 + height ** 2)

# Camera specs
dFOV_deg = 78
dFOV_rad = np.deg2rad(dFOV_deg)

# Compute horizontal and vertical FOV
tan_d = np.tan(dFOV_rad / 2)
tan_h = tan_d * (width / diag_pixels)
tan_v = tan_d * (height / diag_pixels)

hFOV_rad = 2 * np.arctan(tan_h)
vFOV_rad = 2 * np.arctan(tan_v)

print(f"Computed horizontal FOV ≈ {np.rad2deg(hFOV_rad):.2f}°")
print(f"Computed vertical FOV ≈ {np.rad2deg(vFOV_rad):.2f}°")

# Compute focal lengths in pixels
f_x = width / (2 * np.tan(hFOV_rad / 2))
f_y = height / (2 * np.tan(vFOV_rad / 2))
c_x = width / 2
c_y = height / 2

camera_matrix = np.array([[f_x, 0, c_x],
                          [0, f_y, c_y],
                          [0,   0,   1]])
print("Camera matrix:\n", camera_matrix)

# Video processing loop
i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_detection.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks)

    if results.pose_landmarks:
        landmarks_3d = []
        for lm in results.pose_landmarks.landmark:
            x, y, z = lm.x, lm.y, lm.z
            landmarks_3d.append([x, y, z])
        landmarks_3d = np.array(landmarks_3d)

        # Print a landmark every 20 frames
        if i % 30 == 0:
            # Print the angle between left and right shoulders, and their respective elbows
            left_shoulder = landmarks_3d[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks_3d[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_elbow = landmarks_3d[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = landmarks_3d[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

            left_angle = np.arctan2(left_elbow[1] - left_shoulder[1], (left_elbow[0] - left_shoulder[0]) * -1)
            right_angle = np.arctan2(right_elbow[1] - right_shoulder[1], right_elbow[0] - right_shoulder[0])

            # Map angles to be between 0 and 2π
            left_angle = (left_angle + 2 * np.pi) % (2 * np.pi)
            right_angle = (right_angle + 2 * np.pi) % (2 * np.pi)

            # Print angles in degrees
            print(f"Left shoulder angle: {np.rad2deg(left_angle):.2f}°")
            print(f"Right shoulder angle: {np.rad2deg(right_angle):.2f}°")

            # Convert angles to servo positions

            # Left shouler (30 - 195 degrees)
            left_shoulder_angle = 290 - int(np.rad2deg(left_angle))
            left_shoulder_angle = np.clip(left_shoulder_angle, 30, 195)  # Ensure within servo limits

            # Right shoulder (70 - 240 degrees)
            right_shoulder_angle = int(np.rad2deg(right_angle)) - 23
            right_shoulder_angle = np.clip(right_shoulder_angle, 70, 240)

            print(f"Left shoulder servo position: {left_shoulder_angle}")
            print(f"Right shoulder servo position: {right_shoulder_angle}")

    cv2.imshow('Pose Tracking', frame)
    i += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()