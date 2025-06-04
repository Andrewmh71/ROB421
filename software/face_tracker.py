# face_tracker.py
import cv2
import mediapipe as mp

class FaceTracker:
    def __init__(self, frame_width, frame_height):
        self.mp_face = mp.solutions.face_detection
        self.face_detection = self.mp_face.FaceDetection(min_detection_confidence=0.6)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.center_x = frame_width // 2
        self.center_y = frame_height // 2

        # Head Turn (X) - ID 3
        self.servo_x = 120
        self.home_x = 120
        self.min_x = 70
        self.max_x = 160

        # Head Nod (Y) - ID 2
        self.servo_y = 125
        self.home_y = 125
        self.min_y = 110
        self.max_y = 155

        self.lock_width = 125
        self.lock_height = 125
        self.missing_counter = 0
        self.missing_threshold = 30

    def map_range(self, value, in_min, in_max, out_min, out_max):
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def constrain(self, val, min_val, max_val):
        return max(min_val, min(val, max_val))

    def smooth_step(self, target, current, step=2):
        delta = target - current
        if abs(delta) > step:
            delta = step if delta > 0 else -step
        return current + delta

    def get_neck_angles(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb)

        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * self.frame_width)
            y = int(bbox.ymin * self.frame_height)
            w = int(bbox.width * self.frame_width)
            h = int(bbox.height * self.frame_height)
            cx = x + w // 2
            cy = y + h // 2

            dx = cx - self.center_x
            dy = cy - self.center_y

            if abs(dx) > self.lock_width // 2:
                mapped_x = self.map_range(cx, 0, self.frame_width, self.max_x, self.min_x)
                self.servo_x = self.smooth_step(mapped_x, self.servo_x)
                self.servo_x = self.constrain(self.servo_x, self.min_x, self.max_x)

            if abs(dy) > self.lock_height // 2:
                mapped_y = self.map_range(cy, 0, self.frame_height, self.max_y, self.min_y)
                self.servo_y = self.smooth_step(mapped_y, self.servo_y)
                self.servo_y = self.constrain(self.servo_y, self.min_y, self.max_y)

            self.missing_counter = 0
        else:
            self.missing_counter += 1
            if self.missing_counter > self.missing_threshold:
                self.servo_x = self.home_x
                self.servo_y = self.home_y
                self.missing_counter = 0

        return int(self.servo_x), int(self.servo_y)
