import numpy as np
import cv2
from matplotlib import pyplot as plt

cutframe = 1
sumPixelErr = 0
minimum = 233534192 / 2150 #61248906 / 90
width , height = 0, 0
w , h = 10, 10

	
def calcPixelErr(frame, start_x, start_y, end_x, end_y, channel=3) :
	pixelerr = 0
	for x in range (start_x, end_x):
		for y in range (start_y, end_y) :
			for z in range(0, channel) :
				pixelerr += abs(int(frame[x, y][z]) - int(lastframe[x, y][z]))
	return pixelerr

def calcHistPixelErr(frame, lastframe):
	pixelerr = 0
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	hist, bins = np.histogram(gray.flatten(), 256, [0,256])

	p_gray = cv2.cvtColor(lastframe, cv2.COLOR_BGR2GRAY)
	p_hist, p_bins = np.histogram(p_gray.flatten(), 256, [0,256])

	res = hist - p_hist
	
	for i in range(0, len(res)):
		pixelerr += abs(res[i])
		#print 'res[' + str(i) + '] += ' + str(res[i]) + ' , eq = ' + str(pixelerr)

	return pixelerr

def drawHist(frame):
	
	hist, bins = np.histogram(frame.flatten(), 256, [0,256])
	cdf = hist.cumsum()
   	cdf_normalized = cdf * hist.max()/ cdf.max()

   	plt.plot(cdf_normalized, color = 'b')
   	plt.hist(frame.flatten(),256,[0,256], color = 'r')
   	plt.xlim([0,256])
   	plt.legend(('cdf','histogram'), loc = 'upper left')
   	plt.show()

if __name__ == "__main__":

	lastframe = None
	lastStoreframe = lastframe
	nowFrameCount = 0

	cap = cv2.VideoCapture('media/1.mp4')
	width, height, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))	
	cutframe = fps

	#cv2.namedWindow('pre-frame')
	#cv2.moveWindow('pre-frame', 10, 10)
	cv2.namedWindow('frame')
	cv2.moveWindow('frame', 10 + 20, 10 + 20)	

	print('Detect FPS = ' + str(fps))

	while(True):

		nowFrameCount += 1
	
		ret, frame = cap.read()

		if(ret):
			
			if lastframe is None:
				lastframe = frame
				lastStoreframe = lastframe

			elif nowFrameCount % cutframe == 0 :
		
				pixelerr = calcHistPixelErr(frame, lastframe)
				#sumPixelErr += float(pixelerr) / 100.0
				
				# pixelerr = calcPixelErr(frame, 0, 0, w, h)

				print 'nowFrameCount = ' + str(nowFrameCount) + ', pixelerr = ' + str(pixelerr)

				if(pixelerr > minimum) :
					cv2.imwrite('output/' + str(nowFrameCount) + '_1.jpg', lastframe);
					lastStoreframe = lastframe
					
				lastframe = frame

			#cv2.imshow('pre-frame', lastStoreframe)
			cv2.imshow('frame', frame)
				
			k = cv2.waitKey(fps) & 0xFF		
			if(k == ord('q')):
				break
		else:
			print('Done or error when reading file.')
			print sumPixelErr
			break

	cap.release()
	cv2.destroyAllWindows()