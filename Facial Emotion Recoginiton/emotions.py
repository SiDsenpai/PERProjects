import numpy as np
import sys
sys.path.append(r'/Users/siddharthagarwal/Downloads/Project')
import argparse
import matplotlib.pyplot as plt
import cv2
from Model import EmotionNet
from tensorflow.keras.preprocessing.image import img_to_array  # Import img_to_array function

cap = cv2.VideoCapture(0)
facecasc = cv2.CascadeClassifier(r'C:\Users\Saiyajin\Downloads\src\haarcascade_frontalface_default.xml')
net = EmotionNet('model.h5')

while True:
    test_image = cap.read()
    test_image = test_image[1]
    converted_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    faces_detected = facecasc.detectMultiScale(converted_image, scaleFactor=1.3, minNeighbors=5)  # Use converted_image instead of gray

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_image, (x, y), (x+w, y+h), (255, 0, 0))
        roi_gray = converted_image[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        image_pixels = img_to_array(roi_gray)  # Corrected img_to_array function
        image_pixels = np.expand_dims(image_pixels, axis=0)
        image_pixels /= 255

        predictions = net.predict(image_pixels)  # Use the net model for prediction
        max_index = np.argmax(predictions[0])

        emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        emotion_prediction = emotion_detection[max_index]
        label_position = (x, y-10)

        cv2.putText(test_image, emotion_prediction, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)

        resize_image = cv2.resize(test_image, (1000, 700))  # Corrected resizing
        cv2.imshow('Emotion', resize_image)

    else:
        cv2.putText(test_image, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Emotion Detector', test_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()