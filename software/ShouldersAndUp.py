import cv2
import mediapipe as mp

# Initialize MediaPipe pose and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Landmarks for below the waist
below_waist_landmarks = {
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
    mp_pose.PoseLandmark.LEFT_HEEL,
    mp_pose.PoseLandmark.RIGHT_HEEL,
    mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
}

# Face landmarks (forehead, eyes, nose, mouth, ears, etc.)
face_landmarks = list(range(0, 11))  # Rough face landmark indices in MediaPipe Pose

# Set up webcam capture
cap = cv2.VideoCapture(0)

with mp_pose.Pose(static_image_mode=False,
                  model_complexity=1,
                  enable_segmentation=False,
                  min_detection_confidence=0.9,
                  min_tracking_confidence=0.9) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Compute center of face landmarks
            face_points = [landmarks[i] for i in face_landmarks]
            avg_x = sum([pt.x for pt in face_points]) / len(face_points)
            avg_y = sum([pt.y for pt in face_points]) / len(face_points)

            # Draw central face point
            h, w, _ = image.shape
            cv2.circle(image, (int(avg_x * w), int(avg_y * h)), 5, (255, 0, 0), -1)

            # Draw rest of the skeleton except below waist and face
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                if start_idx in below_waist_landmarks or end_idx in below_waist_landmarks:
                    continue
                if start_idx in face_landmarks or end_idx in face_landmarks:
                    continue

                start = landmarks[start_idx]
                end = landmarks[end_idx]

                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))

                cv2.line(image, start_point, end_point, (0, 255, 0), 2)
                cv2.circle(image, start_point, 2, (0, 0, 255), -1)
                cv2.circle(image, end_point, 2, (0, 0, 255), -1)

        cv2.imshow('MediaPipe Pose - Skeleton Tracking (Modified)', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()