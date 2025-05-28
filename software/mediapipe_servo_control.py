import cv2
import serial
import time
import mediapipe as mp

# Serial setup (adjust COM port as needed)
arduino = serial.Serial('COM3', 9600)
time.sleep(2)

# Wait for Arduino ready
while True:
    if arduino.in_waiting:
        line = arduino.readline().decode().strip()
        print("[Arduino]:", line)
        if "Ready for tracking" in line:
            break

# MediaPipe face detection setup
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face.FaceDetection(min_detection_confidence=0.6)

# Camera setup
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Servo configuration
servo_x = 90
servo_y = 125
home_x = 90
home_y = 125
arduino.write(f"{home_x},{home_y}\n".encode())
print(f"[Python] Sent home position X:{home_x}, Y:{home_y}")

# Face lock zone and tracking logic
lock_width = 125
lock_height = 125
center_x = frame_width // 2
center_y = frame_height // 2

# Tracking and frame timing
missing_counter = 0
missing_threshold = 30
frame_delay = 0.1
last_time = time.time()

# Utility functions
def map_range(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def constrain(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def smooth_step(target, current, step=2):
    delta = target - current
    if abs(delta) > step:
        delta = step if delta > 0 else -step
    return current + delta

# Main loop
while True:
    if time.time() - last_time < frame_delay:
        continue
    last_time = time.time()

    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:
        missing_counter = 0
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        x = int(bbox.xmin * frame_width)
        y = int(bbox.ymin * frame_height)
        w = int(bbox.width * frame_width)
        h = int(bbox.height * frame_height)
        cx = x + w // 2
        cy = y + h // 2

        # # MIRRORED X-axis mapping: 180 to 0 (fixes reversed movement)
        # mapped_x = 180 - map_range(cx, 0, frame_width, 180, 0)
        # mapped_y = 180 - map_range(cy, 0, frame_height, 180, 0)

        # # Smooth and constrain
        # servo_x = smooth_step(mapped_x, servo_x)
        # servo_y = smooth_step(mapped_y, servo_y)
        # servo_x = constrain(servo_x, 0, 180)
        # servo_y = constrain(servo_y, 0, 180)
        dx = cx - center_x
        dy = cy - center_y

        # Only update if face is outside the center lock box
        if abs(dx) > lock_width // 2:
            mapped_x = 180 - map_range(cx, 0, frame_width, 180, 0)
            servo_x = smooth_step(mapped_x, servo_x)
            servo_x = constrain(servo_x, 0, 180)

        if abs(dy) > lock_height // 2:
            mapped_y = 180 - map_range(cy, 0, frame_height, 180, 0)
            servo_y = smooth_step(mapped_y, servo_y)
            servo_y = constrain(servo_y, 0, 180)

        # Send to Arduino
        arduino.write(f"{int(servo_x)},{int(servo_y)}\n".encode())
        print(f"[Python → Arduino] X={int(servo_x)}, Y={int(servo_y)}")

        # Draw visuals
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(frame,
                      (center_x - lock_width // 2, center_y - lock_height // 2),
                      (center_x + lock_width // 2, center_y + lock_height // 2),
                      (0, 255, 255), 2)
        cv2.putText(frame, f'Servo X:{int(servo_x)} Y:{int(servo_y)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    else:
        missing_counter += 1
        if missing_counter > missing_threshold:
            # Return to home
            servo_x = home_x
            servo_y = home_y
            arduino.write(f"{servo_x},{servo_y}\n".encode())
            print("[Python] Face lost. Returning to home.")
            missing_counter = 0

    cv2.imshow("Face Lock Tracking", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()
