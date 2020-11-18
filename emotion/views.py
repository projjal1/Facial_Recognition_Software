from django.shortcuts import render,redirect,render_to_response
from django.http import HttpResponse
from django.http import StreamingHttpResponse
import cv2
from emotion.resources.model import FacialExpressionModel
import numpy as np
from sys import platform

try:
    if platform=='linux':
        cam = cv2.VideoCapture(2)
        print("External camera")
    elif platform=='win32':
        cam = cv2.VideoCapture(1)
except:
    print("Internal camera")
    cam = cv2.VideoCapture(0)

cam = cv2.VideoCapture(0)

facec = cv2.CascadeClassifier('emotion/resources/haarcascade_frontalface_default.xml')
model = FacialExpressionModel("emotion/resources/model.json", "emotion/resources/model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = cam.read()  # read the camera frame
        if not success:
            break
        else:
            gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facec.detectMultiScale(gray_fr, 1.3, 5)
            for (x, y, w, h) in faces:
                fc = gray_fr[y:y+h, x:x+w]
                roi = cv2.resize(fc, (48, 48))
                pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
                cv2.putText(frame, pred, (x, y), font, 1, (255, 255, 0), 2)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def video_feed(request):
    #Video streaming route. Put this in the src attribute of an img tag
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def detector(request):
    return render(request,"index.html")