import numpy as np
import cv2
from matplotlib import pyplot as plt

# Note: Use cv2.absDiff to compute the difference between the frames and cv2.sumElems to get the sum of all pixels differences

cutframe = 1

data = { 'frame_info' : [] }

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

	# gray = scale(gray, 0.25, 0.25)
	gray = cv2.resize(gray, None,fx=0.25, fy=0.25, interpolation = cv2.INTER_AREA)
	gray = cv2.GaussianBlur(gray, (9,9), 0.0)

	hist, bins = np.histogram(gray.flatten(), 256, [0,256])

	p_gray = cv2.cvtColor(lastframe, cv2.COLOR_BGR2GRAY)
	p_hist, p_bins = np.histogram(p_gray.flatten(), 256, [0,256])

	res = hist - p_hist

	for i in range(0, len(res)):
		pixelerr += abs(res[i])

	return pixelerr

#
# Draw a RGB histogram of a single frame
#
def drawHist(frame):

	hist, bins = np.histogram(frame.flatten(), 256, [0,256])
	cdf = hist.cumsum()
   	cdf_normalized = cdf * hist.max()/ cdf.max()

   	plt.plot(cdf_normalized, color = 'b')
   	plt.hist(frame.flatten(),256,[0,256], color = 'r')
   	plt.xlim([0,256])
   	plt.legend(('cdf','histogram'), loc = 'upper left')
   	plt.show()

def grabFrameData(source):

	lastframe = None
	nowFrameCount = 0

	cap = cv2.VideoCapture(source)
	width, height, fps , totalFrame= int(cap.get(3)), int(cap.get(4)), int(cap.get(5)), int(cap.get(6))

	#cv2.namedWindow('pre-frame')
	#cv2.moveWindow('pre-frame', 10, 10)
	#cv2.namedWindow('frame')
	#cv2.moveWindow('frame', 10 + 20, 10 + 20)	

	while(True):

		nowFrameCount += 1	# Counts from 1
		ret, frame = cap.read()

		if nowFrameCount == totalFrame - 1 or not ret:
			break

		elif nowFrameCount is 1:
			lastframe = frame
			lastStoreframe = lastframe

		elif nowFrameCount % fps == 0:

			pixelerr = calcHistPixelErr(frame, lastframe)

			data['frame_info'].append({
				'frame' : nowFrameCount,
				'diff' : pixelerr
				})

			#print 'nowFrameCount = ' + str(nowFrameCount) + ', pixelerr = ' + str(pixelerr)

			#if(pixelerr > minimum) :
			#	cv2.imwrite('output/' + str(nowFrameCount) + '_1.jpg', lastframe);
			#	lastStoreframe = lastframe

			lastframe = frame

		#cv2.imshow('pre-frame', lastStoreframe)
		cv2.imshow('frame', frame)

		k = cv2.waitKey(fps) & 0xFF		
		if(k == ord('q')):
			break

	cap.release()
	cv2.destroyAllWindows()

	diff_counts = [frameDataDiff['diff'] for frameDataDiff in data['frame_info']]

	data['stats'] = {
        'num': len(diff_counts),
        'min': np.min(diff_counts),
        'max': np.max(diff_counts),
        'mean': np.mean(diff_counts),
        'median': np.median(diff_counts),
        'std': np.std(diff_counts)
    }

if __name__ == "__main__":

	source = 'media/1.mp4'

	grabFrameData(source)

	limit = 1.85 * data['stats']['std']

	Keyframes = [frame['frame'] for frame in data["frame_info"] if abs(frame['diff'] - data['stats']['mean']) > limit]

	cap = cv2.VideoCapture(source)
	for f in Keyframes:
		cap.set(1, f)
		ret, frame = cap.read()
		cv2.imwrite('output/frame' + str(f) + '.png', frame)
	cap.release() 