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

# Servo offsets
LEFT_SHOULDER_OFFSET = 195
RIGHT_SHOULDER_OFFSET = 70
LEFT_ELBOW_OFFSET = 60
RIGHT_ELBOW_OFFSET = 20
LEFT_BICEP_OFFSET = 20
RIGHT_BICEP_OFFSET = 180

# Other globals
KNEE_MIN_DIST = 0.1
MIN_CONFIDENCE = 0.7

# Initialize robot control
connected = False
try:
    control = JamieControl()
    control.initialize_serial_connection()
    control.load_joint_config('Joint_config.json')
    connected = True
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    connected = False

def angle_between(a, b, c):
    """Returns interior angle ABC (in degrees), using x and y only."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def apply_servo_offset(angle_deg, offset):
    return angle_deg + offset

def apply_offset_inverse(angle_deg, offset):
    """Apply the inverse of the servo offset."""
    return offset - angle_deg

# Wait for biceps to rotate to positions before starting
if connected:
    control.send_joint_command([6, 10], [165, 35], 1)
    time.sleep(2)

left_wrist_below = False
right_wrist_below = False

def process_landmarks(detection_results):
    if not detection_results.pose_landmarks:
        print("No pose landmarks detected.")
        return
    landmarks = detection_results.pose_landmarks[0]

    required = ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "LEFT_HIP",
                "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_HIP"]

    # Extract 2D positions
    ls = np.array([landmarks[GHUM_LANDMARK_NAMES.index("LEFT_SHOULDER")].x,
                   landmarks[GHUM_LANDMARK_NAMES.index("LEFT_SHOULDER")].y])
    le = np.array([landmarks[GHUM_LANDMARK_NAMES.index("LEFT_ELBOW")].x,
                   landmarks[GHUM_LANDMARK_NAMES.index("LEFT_ELBOW")].y])
    lw = np.array([landmarks[GHUM_LANDMARK_NAMES.index("LEFT_WRIST")].x,
                   landmarks[GHUM_LANDMARK_NAMES.index("LEFT_WRIST")].y])
    lh = np.array([landmarks[GHUM_LANDMARK_NAMES.index("LEFT_HIP")].x,
                   landmarks[GHUM_LANDMARK_NAMES.index("LEFT_HIP")].y])

    rs = np.array([landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_SHOULDER")].x,
                   landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_SHOULDER")].y])
    re = np.array([landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_ELBOW")].x,
                   landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_ELBOW")].y])
    rw = np.array([landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_WRIST")].x,
                   landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_WRIST")].y])
    rh = np.array([landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_HIP")].x,
                   landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_HIP")].y])

    command_joints = []
    command_angles = []

    # LEFT SHOULDER
    if landmarks[GHUM_LANDMARK_NAMES.index("LEFT_ELBOW")].visibility > MIN_CONFIDENCE and \
        landmarks[GHUM_LANDMARK_NAMES.index("LEFT_SHOULDER")].visibility > MIN_CONFIDENCE and \
        landmarks[GHUM_LANDMARK_NAMES.index("LEFT_HIP")].visibility > MIN_CONFIDENCE:
        left_shoulder_angle = angle_between(lh, ls, le)
        servo_angle = apply_offset_inverse(left_shoulder_angle, LEFT_SHOULDER_OFFSET)
        command_joints.append(9)
        command_angles.append(servo_angle)

        print(f"Left Shoulder Angle: {left_shoulder_angle:.2f}°")
        print(f"Left Shoulder Servo Angle: {servo_angle:.2f}°")
    
    # RIGHT SHOULDER
    if landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_ELBOW")].visibility > MIN_CONFIDENCE and \
        landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_SHOULDER")].visibility > MIN_CONFIDENCE and \
        landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_HIP")].visibility > MIN_CONFIDENCE:
        right_shoulder_angle = angle_between(rh, rs, re)
        servo_angle = apply_servo_offset(right_shoulder_angle, RIGHT_SHOULDER_OFFSET)
        command_joints.append(5)
        command_angles.append(servo_angle)

        print(f"Right Shoulder Angle: {right_shoulder_angle:.2f}°")
        print(f"Right Shoulder Servo Angle: {servo_angle:.2f}°")

    # LEFT ELBOW
    if landmarks[GHUM_LANDMARK_NAMES.index("LEFT_ELBOW")].visibility > MIN_CONFIDENCE and \
        landmarks[GHUM_LANDMARK_NAMES.index("LEFT_SHOULDER")].visibility > MIN_CONFIDENCE and \
        landmarks[GHUM_LANDMARK_NAMES.index("LEFT_WRIST")].visibility > MIN_CONFIDENCE:
        left_elbow_angle = angle_between(ls, le, lw)
        servo_angle = apply_servo_offset(left_elbow_angle, LEFT_ELBOW_OFFSET)
        command_joints.append(11)
        command_angles.append(servo_angle)

        print(f"Left Elbow Angle: {left_elbow_angle:.2f}°")
        print(f"Left Elbow Servo Angle: {servo_angle:.2f}°")
        
    # RIGHT ELBOW
    if landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_ELBOW")].visibility > MIN_CONFIDENCE and \
        landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_SHOULDER")].visibility > MIN_CONFIDENCE and \
        landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_WRIST")].visibility > MIN_CONFIDENCE:
        right_elbow_angle = angle_between(rs, re, rw)
        servo_angle = apply_servo_offset(right_elbow_angle, RIGHT_ELBOW_OFFSET)
        command_joints.append(7)
        command_angles.append(servo_angle)

        print(f"Right Elbow Angle: {right_elbow_angle:.2f}°")
        print(f"Right Elbow Servo Angle: {servo_angle:.2f}°")

    global left_wrist_below, right_wrist_below
    # LEFT BICEP
    if landmarks[GHUM_LANDMARK_NAMES.index("LEFT_WRIST")].visibility > MIN_CONFIDENCE and \
        landmarks[GHUM_LANDMARK_NAMES.index("LEFT_ELBOW")].visibility > MIN_CONFIDENCE:
        if left_wrist_below != lw[1] > le[1]:
            print("Flipping left bicep")
            left_wrist_below = lw[1] > le[1]
            command_joints.append(10)
            command_angles.append(LEFT_BICEP_OFFSET - 180 if left_wrist_below else LEFT_BICEP_OFFSET)
    
    # RIGHT BICEP
    if landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_WRIST")].visibility > MIN_CONFIDENCE and \
        landmarks[GHUM_LANDMARK_NAMES.index("RIGHT_ELBOW")].visibility > MIN_CONFIDENCE:
        if right_wrist_below != rw[1] > re[1]:
            print("Flipping right bicep")
            right_wrist_below = rw[1] > re[1]
            command_joints.append(8)
            command_angles.append(RIGHT_BICEP_OFFSET - 180 if right_wrist_below else RIGHT_BICEP_OFFSET)

    # Send commands to robot
    if connected and command_joints:
        command_angles = [int(angle) for angle in command_angles]
        print("Sending joint commands:", command_joints, command_angles)
        control.send_joint_command(command_joints, command_angles, 1)
    
    print("\n\n==========================================\n\n")

# Set up mediapipe pose detection
base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionTaskRunningMode.VIDEO
)
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Failed to open video source.")

ft = face_tracker.FaceTracker(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

i = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read frame from camera. Exiting...")
        break

    i += 1
    timestamp = int(time.time() * 1000)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.ascontiguousarray(rgb_frame)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect_for_video(mp_image, timestamp)
    if i % 60 == 0:
        process_landmarks(detection_result)

        if connected:
            angles = ft.get_neck_angles(frame)
            print("Sending neck command", angles)
            control.send_joint_command([3, 2], [angles[0], angles[1]], 1)

    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

control.close_connection()
cap.release()
cv2.destroyAllWindows()
