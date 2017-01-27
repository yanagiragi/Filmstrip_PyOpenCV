import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):

	ret, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	grayblur = cv2.GaussianBlur(gray, (9,9), 0.0)

	cv2.imshow('gray', gray)
	cv2.imshow('blur', gray)

	if(cv2.waitKey(1) & 0xFF == ord('q')):
		break

cap.release()
cv2.destroyAllWindows()