import cv2
import mediapipe as mp
import math
import json

# Mediapipe setup
mp_pose = mp.solutions.pose
mp_pose2 = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
with mp_pose2.Pose(static_image_mode=False,
                  model_complexity=1,
                  enable_segmentation=False,
                  min_detection_confidence=0.9,
                  min_tracking_confidence=0.9) as pose2:
    def calculate_angle(a, b, c):
        """Calculate angle between three points in 3D."""
        def vec(p1, p2):
            return [p2[i] - p1[i] for i in range(3)]

        v1 = vec(b, a)
        v2 = vec(b, c)

        dot_product = sum(v1[i] * v2[i] for i in range(3))
        norm1 = math.sqrt(sum(v1[i] ** 2 for i in range(3)))
        norm2 = math.sqrt(sum(v2[i] ** 2 for i in range(3)))
        
        if norm1 * norm2 == 0:
            return 0.0

        angle = math.acos(dot_product / (norm1 * norm2))
        return math.degrees(angle)

    def get_point(landmark, shape):
        return [
            landmark.x * shape[1],
            landmark.y * shape[0],
            landmark.z * shape[1]
        ]

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

            # Convert the BGR frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # For performance
            
            # Process the image and detect pose
        results = pose2.process(image)

            # Draw the pose annotation on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            shape = frame.shape

            # Get 3D coordinates for relevant joints
            leftShoulder = get_point(lm[mp_pose.PoseLandmark.LEFT_SHOULDER], shape)
            leftElbow = get_point(lm[mp_pose.PoseLandmark.LEFT_ELBOW], shape)
            leftWrist = get_point(lm[mp_pose.PoseLandmark.LEFT_WRIST], shape)

            rightShoulder = get_point(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER], shape)
            rightElbow = get_point(lm[mp_pose.PoseLandmark.RIGHT_ELBOW], shape)
            rightWrist = get_point(lm[mp_pose.PoseLandmark.RIGHT_WRIST], shape)

            # Calculate joint angles
            leftElbow_angle = calculate_angle(leftShoulder, leftElbow, leftWrist)
            leftShoulder_angle = calculate_angle([leftShoulder[0], leftShoulder[1] - 100, leftShoulder[2]], leftShoulder, leftElbow)  # Approx vertical vector
            leftBicep_angle = calculate_angle(leftShoulder, leftElbow, leftWrist) 

            rightElbow_angle = calculate_angle(rightShoulder, rightElbow, rightWrist)
            rightShoulder_angle = calculate_angle([rightShoulder[0], rightShoulder[1] - 100, rightShoulder[2]], rightShoulder, rightElbow)  
            rightBicep_angle = calculate_angle(rightShoulder, rightElbow, rightWrist)

            # Create JSON-like command
            robot_command = {
                "JointAngles": [
                    {"Joint": "LeftShoulder", "Angle": round(leftShoulder_angle, 1)},
                    {"Joint": "LeftBicep", "Angle": round(leftBicep_angle, 1)},
                    {"Joint": "LeftElbow", "Angle": round(leftElbow_angle, 1)},
                    {"Joint": "RightShoulder", "Angle": round(rightShoulder_angle, 1)},
                    {"Joint": "RightBicep", "Angle": round(rightBicep_angle, 1)},
                    {"Joint": "RightElbow", "Angle": round(rightElbow_angle, 1)}
                ]
            }

            if i % 40 == 0:
                print(json.dumps(robot_command, indent=2))
        mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))
        cv2.imshow("Tracking", frame)
        i += 1
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
