from deepface import DeepFace
import cv2
from collections import Counter

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

print("Press 'q' to quit the program.")

# Dictionary to track emotion occurrences
emotion_count = Counter()

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Error: Could not read frame.")
        break

    try:
        # Analyze the current frame for emotions using MTCNN
        results = DeepFace.analyze(
            frame,
            actions=['emotion'],
            detector_backend='mtcnn',  # Use MTCNN for face detection
            enforce_detection=False  # Allow analysis even if no face is detected
        )

        # If results is a list, process the first result
        if isinstance(results, list):
            result = results[0]  # Take the first face detected
        else:
            result = results  # Single face detected

        # Get the dominant emotion
        dominant_emotion = result.get('dominant_emotion', 'Unknown')

        # Add the detected emotion to the counter
        if dominant_emotion != 'Unknown':
            emotion_count[dominant_emotion] += 1

        # Display the emotion on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    except Exception as e:
        print(f"Error analyzing frame: {e}")

    # Display the frame
    cv2.imshow('Real-Time Emotion Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Print the emotion with the maximum count
if emotion_count:
    most_common_emotion, count = emotion_count.most_common(1)[0]
    print(f"The most common emotion detected was '{most_common_emotion}'")
else:
    print("No emotions were detected.")
