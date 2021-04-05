import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import imutils
from imutils.video import VideoStream
 
faceCascade = cv2.CascadeClassifier('mask/resources/haarcascade_frontalface_default.xml')
model = tf.keras.models.load_model("mask/resources/mask_recog2.h5")
prototxtPath = "mask/resources/deploy.prototxt"
weightsPath = "mask/resources/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

def detect_and_predict_mask(frame, faceNet, maskNet):
	# obtaining the dimensions of the frame and then constructing a blob from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# passing the blob through the network and obtaining the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initializing the list of faces, their corresponding locations and the list of predictions from the face mask network
	faces = []
	locs = []
	preds = []

	# looping over the detections
	for i in range(0, detections.shape[2]):
		# obtaining the confidence (i.e., probability) associated with the detection
		confidence = detections[0, 0, i, 2]

		# filtering out weak detections by ensuring the confidence is greater than the minimum confidence
		if confidence > 0.5:
			# computing the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensuring the bounding boxes fall within the dimensions of the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extracting the face ROI, converting it from BGR to RGB channel ordering, resizing it to 224x224 and preprocessing it
			face = frame[startY:endY, startX:endX]
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# adding the face and bounding boxes to the respective lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# making a prediction if at least one face was detected
	if len(faces) > 0:
		# for faster inference performing batch predictions on all faces at the same time rather than one-by-one predictions in the above for loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding locations
	return (locs, preds)


def capture():
	# initialize the video stream
	try:
		vs = cv2.VideoCapture(1)
	except:
		vs = cv2.VideoCapture(0)

	while True:
    	# Capturing frame-by-frame, grabbing the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
		flag,frame = vs.read()
		frame = imutils.resize(frame, width=400)
		
		# detect faces in the frame and determine if they are wearing a
    	# face mask or not
		faces_list=[]
		preds=[]
		(locs, preds) = detect_and_predict_mask(frame, faceNet, model)
    	# loop over the detected face locations and their corresponding
    	# locations
		for (box, pred) in zip(locs, preds):
        	# unpack the bounding box and predictions
			(startX, startY, endX, endY) = box
			(withoutMask,mask, notproper) = pred

        	# determining the class label and color the bounding box and text
			if (mask > withoutMask and mask>notproper):
				label = "Mask"
			elif ( withoutMask > notproper and withoutMask > mask):
				label = "Without Mask"
			else:
				label = "Wear Mask Properly"

			if label == "Mask":
				color = (0, 255, 0)
			elif label=="Without Mask":
				color = (0, 0, 255)
			else:
				color = (255, 140, 0)

        	# including the probability in the label
			label = "{}: {:.2f}%".format(label,max(mask, withoutMask, notproper) * 100)

        	# displaying the label and bounding box rectangle on the output frame
			cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
		# Display the resulting frame
		cv2.imshow('Video', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	vs.release()
	cv2.destroyAllWindows()