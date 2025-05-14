import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose_detection = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.9,
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
        landmarks_2d = []
        for lm in results.pose_landmarks.landmark:
            x, y, z = lm.x, lm.y, lm.z
            landmarks_2d.append([x, y])
        landmarks_2d = np.array(landmarks_2d)

        # Print a landmark every 20 frames
        if i % 30 == 0:
            # Print the angle between left and right shoulders, and their respective elbows
            left_shoulder = landmarks_2d[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks_2d[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_elbow = landmarks_2d[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = landmarks_2d[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            left_wrist = landmarks_2d[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks_2d[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            left_hip = landmarks_2d[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks_2d[mp_pose.PoseLandmark.RIGHT_HIP.value]

            # Find side lengths between elbow, wrist, and shoulders
            l_a = np.linalg.norm(left_shoulder-left_elbow)
            l_b = np.linalg.norm(left_wrist-left_elbow)
            l_c = np.linalg.norm(left_wrist-left_shoulder)

            r_a = np.linalg.norm(right_shoulder-right_elbow)
            r_b = np.linalg.norm(right_wrist-right_elbow)
            r_c = np.linalg.norm(right_wrist-right_shoulder)

            # left_angle_shoulder_initial = np.arctan2(left_elbow[1] - left_shoulder[1], (left_elbow[0] - left_shoulder[0]) * -1)
            # right_angle_shoulder_initial = np.arctan2(right_elbow[1] - right_shoulder[1], right_elbow[0] - right_shoulder[0])
            left_angle_elbow_initial = np.arccos((l_a**2 + l_b**2 - l_c**2) / (2 * l_a * l_b))
            right_angle_elbow_initial = np.arccos((r_a**2 + r_b**2 - r_c**2) / (2 * r_a * r_b))

            l_a = np.linalg.norm(left_shoulder-left_hip)
            l_b = np.linalg.norm(left_shoulder-left_elbow)
            l_c = np.linalg.norm(left_elbow-left_hip)

            r_a = np.linalg.norm(right_shoulder-right_hip)
            r_b = np.linalg.norm(right_shoulder-right_elbow)
            r_c = np.linalg.norm(right_elbow-right_hip)

            left_angle_shoulder_initial = np.arccos((l_a**2 + l_b**2 - l_c**2) / (2 * l_a * l_b))
            right_angle_shoulder_initial = np.arccos((r_a**2 + r_b**2 - r_c**2) / (2 * r_a * r_b))

            # Map angles to be between 0 and 2π
            left_angle_shoulder_initial = (left_angle_shoulder_initial + 2 * np.pi) % (2 * np.pi)
            right_angle_shoulder_initial = (right_angle_shoulder_initial + 2 * np.pi) % (2 * np.pi)
            left_angle_elbow_initial = (left_angle_elbow_initial + 2 * np.pi) % (2*np.pi)
            right_angle_elbow_initial = (right_angle_elbow_initial + 2 * np.pi) % (2*np.pi)
            # Convert angles to servo positions
            
            # Left shoulder (30 - 195 degrees)
            print(f"Left shoulder angle: {np.rad2deg(left_angle_shoulder_initial):.2f}°")
            left_shoulder_angle_servo = 200 - int(np.rad2deg(left_angle_shoulder_initial))
            if(left_shoulder_angle_servo < 30 or left_shoulder_angle_servo > 195):
                print('Left Shoulder Would Collide')
            left_shoulder_angle_servo = np.clip(left_shoulder_angle_servo, 30, 195)  # Ensure within servo limits
            
            # Right shoulder (70 - 240 degrees)
            print(f"Right shoulder angle: {np.rad2deg(right_angle_shoulder_initial):.2f}°")
            right_shoulder_angle_servo = int(np.rad2deg(right_angle_shoulder_initial)) + 60
            if(right_shoulder_angle_servo < 60 or right_shoulder_angle_servo > 240):
                print('Right Shoulder Would Collide')
            right_shoulder_angle_servo = np.clip(right_shoulder_angle_servo, 70, 240)

            left_elbow_angle_servo = int(np.rad2deg(left_angle_elbow_initial))
            if(left_elbow_angle_servo < 0 or left_elbow_angle_servo > 180):
                print('Left ELbow Would Collide')
            left_elbow_angle_servo = np.clip(left_elbow_angle_servo, 0, 180)  # Ensure within servo limits

            right_elbow_angle_servo = int(np.rad2deg(right_angle_elbow_initial))
            if(right_elbow_angle_servo < 0 or right_elbow_angle_servo > 180):
                print('Right Elbow Would Collide')
            right_elbow_angle_servo = np.clip(right_elbow_angle_servo, 0, 180)  # Ensure within servo limits

            print(f"Left elbow servo position: {left_elbow_angle_servo}")
            print(f"Right elbow servo position: {right_elbow_angle_servo}")
            print(f"Left shoulder servo position: {left_shoulder_angle_servo}")
            print(f"Right shoulder servo position: {right_shoulder_angle_servo}")
    
    cv2.imshow('Pose Tracker', frame)
    i += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()