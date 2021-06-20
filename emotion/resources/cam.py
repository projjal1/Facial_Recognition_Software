import cv2
from emotion.resources.model import FacialExpressionModel
import numpy as np

facec = cv2.CascadeClassifier('emotion/resources/haarcascade_frontalface_default.xml')
model = FacialExpressionModel()
font = cv2.FONT_HERSHEY_SIMPLEX

def capture():
    video_capture=cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        _, fr = video_capture.read()
        if not _:
            continue

        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis]) 
            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow('camera',fr)                          
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

