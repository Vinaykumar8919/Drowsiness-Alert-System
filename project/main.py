import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
import time

mixer.init()
sound = mixer.Sound('alarm.wav')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
model = load_model(os.path.join("models", "model.h5"))

lbl=['Close', 'Open']

path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score_open = 0  # Variable to keep track of open eye score
score_close = 0  # Variable to keep track of close eye score
alarm_playing = False  # To keep track of alarm state

while(True):
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, minNeighbors=3, scaleFactor=1.1, minSize=(25, 25))
    eyes = eye_cascade.detectMultiScale(gray, minNeighbors=1, scaleFactor=1.1)

    cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)

    for (x, y, w, h) in eyes:
        eye_region = frame[y:y+h, x:x+w]
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        resized_eye = cv2.resize(gray_eye, (80, 80))  # Resize to match the model input
        resized_eye_rgb = np.repeat(resized_eye[:, :, np.newaxis], 3, axis=2)

        normalized_eye = resized_eye_rgb / 255.0
        reshaped_eye = normalized_eye.reshape(80, 80, 3)
        expanded_eye = np.expand_dims(reshaped_eye, axis=0)  # Add batch dimension
        prediction = model.predict(expanded_eye)

        if prediction[0][0] > 0.30:
            
            cv2.putText(frame,   str(prediction[0][0]), (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            score_close += 1
            score_open = 0  # Reset open eye score
            if score_close >5 and not alarm_playing:
                try:
                    sound.play()
                    alarm_playing = True
                except:
                    pass
        else:
            score_open += 1
            score_close = 0  # Reset close eye score
            if alarm_playing and score_open>5:
                try:
                    sound.stop()
                    alarm_playing = False
                except:
                    pass
            
            cv2.putText(frame, str(prediction[0][0]), (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('frame', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
