import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
from read_json import JamieControl
import time
from mediapipe.framework.formats import image_format_pb2
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
import math

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

# Initialize robot control
connected = False
try:
    control = JamieControl()
    control.initialize_serial_connection()
    control.load_joint_config('Joint_config.json')
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    connected = False

import numpy as np

def cartesian_to_spherical(x, y, z):
    r = math.sqrt(x**2 + y**2 + z**2)
    theta = math.atan2(z, x)
    phi = math.asin(z / r) if r != 0 else 0
    return [r, math.degrees(theta), math.degrees(phi)]

def elbow_angle(shoulder, elbow, wrist):
    a = shoulder - elbow
    b = wrist - elbow
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

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
first = True
ref_vector = np.array([0, 0, 0])
i = 0

def process_landmarks(detection_results, output_image, timestamp):
    if i % 20 != 0:
        return

    landmarks = detection_results.pose_landmarks[0]
    print("\n\n\n== GHUM Model Output (pose_landmarks) ==")

    for idx, lm in enumerate(landmarks):
        if lm.visibility > 0.85 and lm.presence > 0.4 and idx in [11, 12, 13, 14, 15, 16]:
            joint = GHUM_LANDMARK_NAMES[idx]
            print(f"{idx:2d}: {joint:20s} | x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f} | "
                f"vis={lm.visibility:.3f}, pres={lm.presence:.3f}")
    
    # Continue if shoulder, elbow, wrist or hip are not visible
    if (landmarks[GHUM_LANDMARK_NAMES.index("LEFT_SHOULDER")].visibility < 0.8 or
        landmarks[GHUM_LANDMARK_NAMES.index("LEFT_ELBOW")].visibility < 0.8 or
        landmarks[GHUM_LANDMARK_NAMES.index("LEFT_WRIST")].visibility < 0.8 or
        landmarks[GHUM_LANDMARK_NAMES.index("LEFT_HIP")].visibility < 0.8 or
        landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_SHOULDER")].visibility < 0.8 or
        landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_ELBOW")].visibility < 0.8 or
        landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_WRIST")].visibility < 0.8 or
        landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_HIP")].visibility < 0.8):
        return
    
    print("Saw visible landmarks, processing...")

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

    # if first:
    #     ref_vector = compute_ref_vector(left_shoulder, left_elbow, left_wrist)
    #     first = False
    
    # Calculate angles
    left_elbow_angle = elbow_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = elbow_angle(right_shoulder, right_elbow, right_wrist)

    # Reference vector is the negative z-axis
    # (towards camera is negative z)
    left_bicep_rotation = bicep_rotation(left_shoulder, left_elbow, left_wrist)
    right_bicep_rotation = bicep_rotation(right_shoulder, right_elbow, left_wrist)
    left_shoulder_rot = cartesian_to_spherical(left_shoulder[0], left_shoulder[1], left_shoulder[2])
    right_shoulder_rot = cartesian_to_spherical(right_shoulder[0], right_shoulder[1], right_shoulder[2])

    # Print angles
    print(f"Left elbow angle: {np.rad2deg(left_elbow_angle):.2f}°")
    print(f"Right elbow angle: {np.rad2deg(right_elbow_angle):.2f}°")
    print(f"Left shoulder abduction: {left_shoulder_rot[1]:.2f}°")
    print(f"Right shoulder abduction: {right_shoulder_rot[1]:.2f}°")
    print(f"Left chest flexion: {left_shoulder_rot[2]:.2f}°")
    print(f"Right chest flexion: {right_shoulder_rot[2]:.2f}°")

# Set up mediapipe pose detection
base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionTaskRunningMode.LIVE_STREAM,
    result_callback=process_landmarks
)
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Failed to open video source.")
timestamp = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read frame from camera. Exiting...")
        break

    i += 1
    timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rgb_frame = np.ascontiguousarray(rgb_frame)

    # Wrap in MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Send to detector
    detector.detect_async(mp_image, timestamp)

    # Optional display
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

control.close_connection()
cap.release()
cv2.destroyAllWindows()
