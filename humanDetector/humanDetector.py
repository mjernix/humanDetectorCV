import cv2
from random import randrange

#load pre-trained data
trainedFaceData = cv2.CascadeClassifier('recognitionData/haarcascade_frontalface_default.xml')
trainedEyeData = cv2.CascadeClassifier('recognitionData/haarcascade_eye.xml')

#Enable the use of main camera(selected by the VideoCapture argument)

webcam = cv2.VideoCapture(0)


#go over frames 
while True:
	#read frame
	succesfulFrameRead, frame = webcam.read()

	#Greyscale image
	greyscaledImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#detect faces
	faceCoordinates = trainedFaceData.detectMultiScale(greyscaledImg)
	eyeCoordinates = trainedEyeData.detectMultiScale(greyscaledImg)

	#draw rectangles for faces(for loop for multiple faces)
	for i in faceCoordinates:
		(x, y, w, h) = i
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255,0), (2))

	#draw faces for bodies
	for i in eyeCoordinates:
		(x, y, w, h) = i
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), (1)) 
	
	#show image
	cv2.imshow('Human Detector', frame) 
	#stop the program from closing immediately
	key = cv2.waitKey(1)

	#quit app if q is pressed
	if key == 81 or key == 113:
		break

 

#release VideoCapture
webcam.release()

print("Code finished running without errors.")