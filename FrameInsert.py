import numpy as np
import cv2
from matplotlib import pyplot as plt

alpha = .5
beta = 1.0 - alpha

def run(source):
	
	nowFrameCount = 0

	cap = cv2.VideoCapture(source)
	width, height, fps , totalFrame= int(cap.get(3)), int(cap.get(4)), int(cap.get(5)), int(cap.get(6))

	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('output.avi',fourcc, fps * 2, (width, height))
	
	while(True):

		nowFrameCount += 1	# Counts from 1
		ret, frame = cap.read()

		if nowFrameCount == totalFrame - 1 or not ret:
			out.write(lastframe)
			break
		elif nowFrameCount is 1:
			lastframe = frame
			continue
		else:
			middleframe = lastframe
			cv2.addWeighted(lastframe, alpha, frame, beta, 0.0, middleframe)
			#middleframe = (frame + lastframe)
			out.write(lastframe)
			out.write(middleframe)
			
			#cv2.imwrite(str(nowFrameCount) + 'l.png', lastframe)
			#cv2.imwrite(str(nowFrameCount) + 'm.png', middleframe)
			#cv2.imwrite(str(nowFrameCount) + 'n.png', frame)
			#if nowFrameCount == 20:
			#	break
			lastframe = frame

		cv2.imshow('frame', frame)
			
		k = cv2.waitKey(fps) & 0xFF		
		if(k == ord('q')):
			break

	cap.release()
	out.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":

	source = 'media/1.mp4'
	run(source)