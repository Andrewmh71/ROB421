import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
from read_json import JamieControl
import time
from mediapipe.framework.formats import image_format_pb2
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
import math
import face_tracker

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
    connected = True
    connected = True
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    connected = False

import numpy as np

def zero(toZero, Zero):
    toZero[0] = toZero[0]-Zero[0]
    toZero[1] = toZero[1]-Zero[1]
    toZero[2] = toZero[2]-Zero[2]
    return toZero

# def cartesian_to_spherical(x, y, z):
#     r = math.sqrt(x**2 + y**2 + z**2)
#     theta = math.atan2(z, x)
#     phi = math.asin(z / r) if r != 0 else 0
#     return [r, math.degrees(theta), math.degrees(phi)]

def cartesian_to_spherical(joint):
    r = math.sqrt(joint[0]**2 + joint[1]**2 + joint[2]**2)
    theta = math.atan2(joint[2], joint[1])
    phi = math.asin(joint[2] / r) if r != 0 else 0
    return [r, math.degrees(theta), math.degrees(phi)]

def elbow_angle(shoulder, elbow, wrist):
    a = shoulder - elbow
    b = wrist - elbow
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

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
    left_shoulder_rot = cartesian_to_spherical(np.array(zero(left_elbow,left_shoulder)))
    right_shoulder_rot = cartesian_to_spherical(np.array(zero(right_elbow,right_shoulder)))
    
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

ft = face_tracker.FaceTracker(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    if connected and i % 60 == 0:
        angles = ft.get_neck_angles(frame)
        print("Sending neck command", angles)
        control.send_joint_command([3, 2], [angles[0], angles[1]], 1)

    # # Send to detector
    # detector.detect_async(mp_image, timestamp)

    # Optional display
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

control.close_connection()
cap.release()
cv2.destroyAllWindows()
