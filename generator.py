import cv2
import numpy as np
import random

'''
A Generator that produces sets of training features and labels
* video:  A list of video frames
* speeds: A list of corresponding speeds
* batch_size: number of samples to produce for each batch
* sequence_length: number of images that model consider for each prediction
'''
def data_generator(video, speeds, batch_size, sequence_length):
	while True:
		x = []
		y = []
		while len(x) < batch_size:
			frame_num = random.randrange(sequence_length,len(video))
			frames = []
			for i in range(sequence_length-1,-1,-1):
				frame = video[frame_num-i]
				frames.append(frame)
				x.append(video[frame_num-sequence_length:frame_num])
				y.append(speeds[frame_num])
			yield np.array(x), np.array(y)