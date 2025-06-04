import cv2
from deepface import DeepFace

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access the camera.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Display the frame
    cv2.imshow("Webcam Feed", frame)

    try:
        # Analyze the frame for emotion
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # Print the dominant emotion
        dominant_emotion = result[0]['dominant_emotion']
        print(f"Expression: {dominant_emotion}")
    except Exception as e:
        print(f"Analysis failed: {e}")

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
