import numpy as np
import sys
sys.path.append(r'/Users/siddharthagarwal/Downloads/Project')
import cv2
from Model import EmotionNet

#Load pre-trained emotion detection model
emotion_model = EmotionNet()

# Initialize video capture object for webcam
cap = cv2.VideoCapture(0)  # Change to 1 if using an external webcam

# Load Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # For each detected face, predict emotion and draw rectangle and text
    for (x, y, w, h) in faces:
        # Extract region of interest (ROI) for face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Preprocess ROI for emotion prediction
        cropped_img = cv2.resize(roi_gray, (48, 48))
        cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)

        # Predict emotion
        prediction = emotion_model.predict(cropped_img)

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Write predicted emotion label on the frame
        cv2.putText(frame, prediction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()