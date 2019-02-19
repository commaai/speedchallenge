import cv2
import numpy as np
import random


'''
	A Generator that produces sets of training features and labels
	* cap: cv2 videocapture object for video
	* size: number of frames in the video
	* batch_size: number of samples to produce for each batch
	* sequence_length: number of images that model consider for each prediction
	* shape: tuple with desired image sizes (height, width)
	* speeds: text file containing speeds at given times
'''
def data_generator(cap, size, batch_size, sequence_length, shape,speeds=None):
	# Initialize speed data
	if speeds != None:
		with open('data/train.txt') as f:
			speeds = f.readlines()
		speeds = np.array([float(x.strip()) for x in speeds]) 
	else:
		speeds = np.empty(size)

	while True:
		x = []
		y = []
		while len(x) < batch_size:
			frame_num = random.randrange(0,size)
			frames = []
			for i in range(sequence_length-1,-1,-1):
				# Get a singgle frame, 1 is the ordinal value of CV_CAP_PROP_POS_FRAMES
				cap.set(1, frame_num-i)
				ret, frame = cap.read()
				frame = cv2.resize(frame,shape)
				frames.append(frame)
			x.append(frames)
			y.append(speeds[frame_num])
		x = np.array(x)
		y = np.array(y)
		yield x, y

