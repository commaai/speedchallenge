import cv2
import numpy as np
import random
from opticalflow import denseflow

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
	print("Processing speeds")
	if speeds != None:
		with open('data/train.txt') as f:
			speeds = f.readlines()
			speeds = np.array([float(x.strip()) for x in speeds]) 
	else:
		speeds = np.empty(size)

	print("Processing video")
	video = denseflow(cap, size, shape)

	'''
	video = np.zeros((size,shape[1],shape[0],3))
		# Read until video is completed
	for i in range(size):
		# Capture frame-by-frame
		ret, frame = cap.read()
		if ret is True:
			video[i] = cv2.resize(frame,shape) / 255
	'''
	cap.release()

	while True:
		x = []
		y = []
		while len(x) < batch_size:
			frame_num = random.randrange(sequence_length,size)
			frames = []
			for i in range(sequence_length-1,-1,-1):
				frame = video[frame_num-i]
				frames.append(frame)
				x.append(video[frame_num-sequence_length:frame_num])
				y.append(speeds[frame_num])
			yield np.array(x), np.array(y)