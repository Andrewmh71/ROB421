import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
from read_json import JamieControl
import time
from mediapipe.framework.formats import image_format_pb2
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

GHUM_LANDMARK_NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR",
    "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER",
    "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST",
    "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX",
    "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP",
    "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE",
    "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX"
]

# Set up mediapipe pose detection
base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionTaskRunningMode.VIDEO
)
detector = vision.PoseLandmarker.create_from_options(options)

# Open video (0 for webcam, or path to video file)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Failed to open video source.")

# Initialize robot control
connected = False
try:
    control = JamieControl()
    control.initialize_serial_connection()
    control.load_joint_config('Joint_config.json')
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    connected = False

if connected: time.sleep(2)  # Wait for servos to go to config positions

# # Initialize MediaPipe Pose
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# pose_detection = mp_pose.Pose(
#     static_image_mode=False,
#     min_detection_confidence=0.9,
#     enable_segmentation=False,
#     model_complexity=1
# )

# # Read one frame to get dimensions
# ret, frame = cap.read()
# if not ret:
#     raise RuntimeError("Failed to read from video source.")

# height, width, _ = frame.shape
# diag_pixels = np.sqrt(width ** 2 + height ** 2)

# # Camera specs
# dFOV_deg = 78
# dFOV_rad = np.deg2rad(dFOV_deg)

# # Compute horizontal and vertical FOV
# tan_d = np.tan(dFOV_rad / 2)
# tan_h = tan_d * (width / diag_pixels)
# tan_v = tan_d * (height / diag_pixels)

# hFOV_rad = 2 * np.arctan(tan_h)
# vFOV_rad = 2 * np.arctan(tan_v)

# print(f"Computed horizontal FOV ≈ {np.rad2deg(hFOV_rad):.2f}°")
# print(f"Computed vertical FOV ≈ {np.rad2deg(vFOV_rad):.2f}°")

# # Compute focal lengths in pixels
# f_x = width / (2 * np.tan(hFOV_rad / 2))
# f_y = height / (2 * np.tan(vFOV_rad / 2))
# c_x = width / 2
# c_y = height / 2

# camera_matrix = np.array([[f_x, 0, c_x],
#                           [0, f_y, c_y],
#                           [0,   0,   1]])
# print("Camera matrix:\n", camera_matrix)

import numpy as np

def elbow_angle(shoulder, elbow, wrist):
    a = shoulder - elbow
    b = wrist - elbow
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))  # radians

def compute_ref_vector(shoulder, elbow, wrist):
    axis = np.linalg.norm(elbow - shoulder)
    raw = wrist - elbow
    ref_proj = raw - np.dot(raw, axis) * axis
    print(f"ref_proj: {ref_proj}")
    return np.linalg.norm(ref_proj)

def bicep_rotation(shoulder, elbow, wrist, reference=np.array([0, 0, -1])):
    """
    Computes the signed twist angle (in radians) of the forearm around the upper arm,
    relative to a fixed reference direction in the orthogonal plane.

    Parameters:
        shoulder: np.array of shape (3,)
        elbow: np.array of shape (3,)
        wrist: np.array of shape (3,)
        reference: np.array of shape (3,), default is negative Z (camera-facing)

    Returns:
        angle: float, twist angle in radians
    """
    # Vector from shoulder to elbow = upper arm
    u = elbow - shoulder
    u_norm = u / np.linalg.norm(u)  # This is the rotation axis

    # Vector from elbow to wrist = forearm
    f = wrist - elbow

    # Project reference and forearm into plane orthogonal to upper arm
    def project_onto_plane(v, normal):
        return v - np.dot(v, normal) * normal

    f_proj = project_onto_plane(f, u_norm)
    r_proj = project_onto_plane(reference, u_norm)

    # Normalize projections
    f_proj_norm = np.linalg.norm(f_proj)
    r_proj_norm = np.linalg.norm(r_proj)
    if f_proj_norm < 1e-8 or r_proj_norm < 1e-8:
        raise ValueError("Projection too small; vectors are nearly aligned with upper arm.")

    f_unit = f_proj / f_proj_norm
    r_unit = r_proj / r_proj_norm

    # Signed angle from reference to forearm in plane
    angle = np.arctan2(
        np.dot(np.cross(r_unit, f_unit), u_norm),  # signed component
        np.dot(r_unit, f_unit)                     # cosine of angle
    )

    return angle

def shoulder_abduction(shoulder, elbow, hip):
    arm_vec = elbow - shoulder
    torso_vec = hip - shoulder

    # Project arm_vec onto torso's left-right + up-down plane
    # We'll assume camera is facing front, so forward (Z) is ignored
    arm_proj = arm_vec.copy()
    arm_proj[2] = 0  # zero out Z
    torso_proj = torso_vec.copy()
    torso_proj[2] = 0

    arm_proj /= np.linalg.norm(arm_proj)
    torso_proj /= np.linalg.norm(torso_proj)

    cos_theta = np.dot(arm_proj, torso_proj)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))  # radians

def chest_flexion(shoulder, elbow, hip):
    arm_vec = elbow - shoulder
    torso_vec = hip - shoulder

    # Project onto Y-Z plane (ignore left-right X)
    arm_proj = arm_vec.copy()
    arm_proj[0] = 0
    torso_proj = torso_vec.copy()
    torso_proj[0] = 0

    arm_proj /= np.linalg.norm(arm_proj)
    torso_proj /= np.linalg.norm(torso_proj)

    cos_theta = np.dot(arm_proj, torso_proj)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))  # radians

# Wait for biceps to rotate to positions before starting
if connected:
    control.send_joint_command([6, 10], [165, 35], 1)

    time.sleep(2) 

# Video processing loop
i = 0
timestamp = 0
first = True
ref_vector = np.array([0, 0, 0])


while True:
    ret, frame = cap.read()
    if not ret:
        break

    i += 1

    cv2.imshow('Pose Tracker', frame)

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.ascontiguousarray(rgb_frame)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    detection_results = detector.detect_for_video(mp_image, timestamp)

    timestamp += int(1e6 / cap.get(cv2.CAP_PROP_FPS))

    if detection_results.pose_landmarks and i % 10 == 0:
        landmarks = detection_results.pose_landmarks[0]
        print("\n\n\n== GHUM Model Output (pose_landmarks) ==")
        for idx, lm in enumerate(landmarks):
            if lm.visibility > 0.85 and lm.presence > 0.4 and idx in [11, 12, 13, 14, 15, 16, 23, 24]:
                joint = GHUM_LANDMARK_NAMES[idx]
                print(f"{idx:2d}: {joint:20s} | x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f} | "
                    f"vis={lm.visibility:.3f}, pres={lm.presence:.3f}")
        
        # Continue if shoulder, elbow, wrist or hip are not visible
        if (landmarks[GHUM_LANDMARK_NAMES.index("LEFT_SHOULDER")].visibility < 0.5 or
            landmarks[GHUM_LANDMARK_NAMES.index("LEFT_ELBOW")].visibility < 0.5 or
            landmarks[GHUM_LANDMARK_NAMES.index("LEFT_WRIST")].visibility < 0.5 or
            landmarks[GHUM_LANDMARK_NAMES.index("LEFT_HIP")].visibility < 0.5 or
            landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_SHOULDER")].visibility < 0.5 or
            landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_ELBOW")].visibility < 0.5 or
            landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_WRIST")].visibility < 0.5 or
            landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_HIP")].visibility < 0.5):
            continue
                
        # Form appropriate numpy arrays
        left_shoulder = np.array([landmarks[GHUM_LANDMARK_NAMES.index("LEFT_SHOULDER")].x,
                                    landmarks[GHUM_LANDMARK_NAMES.index("LEFT_SHOULDER")].y,
                                    landmarks[GHUM_LANDMARK_NAMES.index("LEFT_SHOULDER")].z])
        right_shoulder = np.array([landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_SHOULDER")].x,
                                    landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_SHOULDER")].y,
                                    landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_SHOULDER")].z])
        left_elbow = np.array([landmarks[GHUM_LANDMARK_NAMES.index("LEFT_ELBOW")].x,
                                landmarks[GHUM_LANDMARK_NAMES.index("LEFT_ELBOW")].y,
                                landmarks[GHUM_LANDMARK_NAMES.index("LEFT_ELBOW")].z])
        right_elbow = np.array([landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_ELBOW")].x,
                                landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_ELBOW")].y,
                                landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_ELBOW")].z])
        left_wrist = np.array([landmarks[GHUM_LANDMARK_NAMES.index("LEFT_WRIST")].x,
                                landmarks[GHUM_LANDMARK_NAMES.index("LEFT_WRIST")].y,
                                landmarks[GHUM_LANDMARK_NAMES.index("LEFT_WRIST")].z])
        right_wrist = np.array([landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_WRIST")].x,
                                landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_WRIST")].y,
                                landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_WRIST")].z])
        left_hip = np.array([landmarks[GHUM_LANDMARK_NAMES.index("LEFT_HIP")].x,
                                landmarks[GHUM_LANDMARK_NAMES.index("LEFT_HIP")].y,
                                landmarks[GHUM_LANDMARK_NAMES.index("LEFT_HIP")].z])
        right_hip = np.array([landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_HIP")].x,
                                landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_HIP")].y,
                                landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_HIP")].z])

        if first:
            ref_vector = compute_ref_vector(left_shoulder, left_elbow, left_wrist)
            first = False
         
        # Calculate angles
        left_elbow_angle = elbow_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = elbow_angle(right_shoulder, right_elbow, right_wrist)

        # Reference vector is the negative z-axis
        # (towards camera is negative z)
        left_bicep_rotation = bicep_rotation(left_shoulder, left_elbow, left_wrist)
        right_bicep_rotation = bicep_rotation(right_shoulder, right_elbow, right_wrist)
        left_shoulder_abduction = shoulder_abduction(left_shoulder, left_elbow, left_hip)
        right_shoulder_abduction = shoulder_abduction(right_shoulder, right_elbow, right_hip)
        left_chest_flexion = chest_flexion(left_shoulder, left_elbow, left_hip)
        right_chest_flexion = chest_flexion(right_shoulder, right_elbow, right_hip)

        # Print angles
        print(f"Left elbow angle: {np.rad2deg(left_elbow_angle):.2f}°")
        print(f"Right elbow angle: {np.rad2deg(right_elbow_angle):.2f}°")
        # print(f"Left bicep rotation: {np.rad2deg(left_bicep_rotation):.2f}°")
        # print(f"Right bicep rotation: {np.rad2deg(right_bicep_rotation):.2f}°")
        # print(f"Left shoulder abduction: {np.rad2deg(left_shoulder_abduction):.2f}°")
        # print(f"Right shoulder abduction: {np.rad2deg(right_shoulder_abduction):.2f}°")
        # print(f"Left chest flexion: {np.rad2deg(left_chest_flexion):.2f}°")
        # print(f"Right chest flexion: {np.rad2deg(right_chest_flexion):.2f}°")

    #         # Convert angles to servo positions
            
    #         # Left shoulder (30 - 195 degrees)
    #         print(f"Left shoulder angle: {np.rad2deg(left_angle_shoulder_initial):.2f}°")
    #         left_shoulder_angle_servo = 200 - int(np.rad2deg(left_angle_shoulder_initial))
    #         if(left_shoulder_angle_servo < 30 or left_shoulder_angle_servo > 195):
    #             print('Left Shoulder Would Collide')
    #         left_shoulder_angle_servo = np.clip(left_shoulder_angle_servo, 30, 195)  # Ensure within servo limits
            
    #         # Right shoulder (70 - 240 degrees)
    #         print(f"Right shoulder angle: {np.rad2deg(right_angle_shoulder_initial):.2f}°")
    #         right_shoulder_angle_servo = int(np.rad2deg(right_angle_shoulder_initial)) + 60
    #         if(right_shoulder_angle_servo < 60 or right_shoulder_angle_servo > 240):
    #             print('Right Shoulder Would Collide')
    #         right_shoulder_angle_servo = np.clip(right_shoulder_angle_servo, 70, 240)

    #         left_elbow_angle_servo = int(np.rad2deg(left_angle_elbow_initial))
    #         if(left_elbow_angle_servo < 0 or left_elbow_angle_servo > 180):
    #             print('Left ELbow Would Collide')
    #         left_elbow_angle_servo = np.clip(left_elbow_angle_servo, 0, 180)  # Ensure within servo limits

    #         right_elbow_angle_servo = int(np.rad2deg(right_angle_elbow_initial))
    #         if(right_elbow_angle_servo < 0 or right_elbow_angle_servo > 180):
    #             print('Right Elbow Would Collide')
    #         right_elbow_angle_servo = np.clip(right_elbow_angle_servo, 0, 180)  # Ensure within servo limits

    #         print(f"Left elbow servo position: {left_elbow_angle_servo}")
    #         print(f"Right elbow servo position: {right_elbow_angle_servo}")
    #         print(f"Left shoulder servo position: {left_shoulder_angle_servo}")
    #         print(f"Right shoulder servo position: {right_shoulder_angle_servo}")

    #         if connected:
    #             control.send_joint_command([9, 11, 5, 7], 
    #                                     [left_shoulder_angle_servo, left_elbow_angle_servo,
    #                                         right_shoulder_angle_servo, right_elbow_angle_servo], 1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

control.close_connection()
cap.release()
cv2.destroyAllWindows()