

import cv2
from deepface import DeepFace
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras

# Path to the authorized user's image
AUTHORIZED_IMAGE = "authorized.jpg"

# Temporary file for captured webcam image
TEMP_CAPTURE = "captured.jpg"

# Capture image from webcam
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return False

    print("📸 Press SPACE to capture, ESC to exit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        cv2.imshow("Capture Image", frame)
        key = cv2.waitKey(1)

        if key % 256 == 27:  # ESC key
            print("🚪 Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            return False
        elif key % 256 == 32:  # SPACE key
            cv2.imwrite(TEMP_CAPTURE, frame)
            print(f"✅ Image captured and saved to {TEMP_CAPTURE}")
            break

    cap.release()
    cv2.destroyAllWindows()
    return True

# Verify identity using DeepFace
def verify_user():
    try:
        result = DeepFace.verify(img1_path=AUTHORIZED_IMAGE,
                                 img2_path=TEMP_CAPTURE,
                                 enforce_detection=True)

        if result["verified"]:
            print("✅ Access granted: Authorized user.")
            #Set up run motion capture####################################################################################################################################
        else:
            print("❌ Access denied: Unauthorized user.")
    except Exception as e:
        print(f"❌ Error during verification: {e}")

# Main logic
if __name__ == "__main__":
    if not os.path.exists(AUTHORIZED_IMAGE):
        print(f"❌ Authorized image '{AUTHORIZED_IMAGE}' not found.")
    elif capture_image():
        verify_user()
        os.remove(TEMP_CAPTURE)  # Clean up


