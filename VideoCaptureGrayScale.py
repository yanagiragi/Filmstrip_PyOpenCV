import numpy as np
import cv2

def calcPixelErr(frame, start_x, start_y, end_x, end_y, channel=3) :
	pixelerr = 0
	for x in range (start_x, end_x):
		for y in range (start_y, end_y) :
			for z in range(0, channel) :
				pixelerr += abs(int(frame[x, y][z]) - int(lastframe[x, y][z]))
	return pixelerr

if __name__ == "__main__":

	nowframe = 0

	r, h, c, w = 250,90,400,125

	cap = cv2.VideoCapture('media/1.mp4')
	#w , h = int(cap.get(3)), int(cap.get(4))
	w , h = 10, 10
	fps = int(cap.get(5))

	cutframe = 20
	minimum = 0

	lastframe = None

	print('fps = ' + str(fps))

	while(True):

		nowframe += 1
	
		ret, frame = cap.read()

		if(ret):
			if lastframe is None:
				lastframe = frame
			elif nowframe % cutframe == 0 :
				pixelerr = calcPixelErr(frame, 0, 0, w, h)

				print 'nowframe = ' + str(nowframe) + ', pixelerr = ' + str(pixelerr)

				if(pixelerr <= minimum) :
					cv2.imwrite('output/' + str(nowframe) + '_1.jpg', lastframe);
					cv2.imwrite('output/' + str(nowframe) + '_2.jpg', frame);
				lastframe = frame
				#cv2.imwrite('output/' + str(nowframe) + '_0.jpg', frame);

			cv2.imshow('frame', frame)

			#cv2.imshow('frame', cv2.rectangle(frame, (0,0), (w,h), (0,0,0)))
			
			k = cv2.waitKey(fps) & 0xFF		
			if(k == ord('q')):
				break
		else:
			print('Done or error when reading file.')
			break

	cap.release()
	cv2.destroyAllWindows()