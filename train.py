import cv2
import matplotlib.pyplot as plt
from generator import data_generator, prediction_generator
from tensorflow.keras.layers import Input, Dense, Flatten, Conv3D, MaxPooling3D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, RMSprop
from opticalflow import denseflow
import numpy as np

batch_size = 16
sequence_length = 5
epochs = 20
split = .9

print("Processing speeds.")
with open('data/train.txt') as f:
	speeds = f.readlines()
	speeds = np.array([float(x.strip()) for x in speeds]) 

print("Processing video.")
cap = cv2.VideoCapture('./data/train.mp4')
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
  exit()

# Get the number of frames, 7 is the ordinal value of CV_CAP_PROP_FRAME_COUNT
video_size = int(cap.get(7))

# Get the video dimensions, 3 is the ordinal value of CV_CAP_PROP_FRAME_WIDTH, 4 is CV_CAP_PROP_FRAME_HEIGHT
# Also resize them because these images are too big
width = int(cap.get(3) / 4)
height = int(cap.get(4) / 4)

try:
	with np.load('./data/asdtrain_op.npz') as data:
		video = data['arr_0']
	print("Found preprocessed video, loading")
except:
	print("Could not find preprocessed video, creating it now")
	video = denseflow(cap, video_size, (width,height))

cap.release()
print(video_size)
print(len(video))
#p = np.random.permutation(video_size)
#video = video[p]
#speeds = speeds[p]

train_gen = data_generator(video[:int(video_size*split)], speeds[:int(video_size*split)], batch_size, sequence_length)
val_gen = data_generator(video[int(video_size*split):], speeds[int(video_size*split):], batch_size, sequence_length)
pred_gen = prediction_generator(video, sequence_length)

# Will return a feature and label set.	
# Features are a list of image sequences in the form: (sequence_length, img_height, img_width, dimensions)
inputs = Input((sequence_length,height,width,3))

# A convolution being applied to each image seperately
x = Conv3D(32,(1,3,3),strides=(1,2,2),activation='relu')(inputs)
x = BatchNormalization()(x)
x = Conv3D(32,(1,3,3),strides=(1,2,2),activation='relu')(x)
x = BatchNormalization()(x)
x = Conv3D(32,(1,3,3),strides=(1,2,2),activation='relu')(x) 
x = BatchNormalization()(x)

# A convolution across all images together
x = Conv3D(16,(3,1,1),strides=(1,1,1),activation='relu')(x)
x = BatchNormalization()(x)
#x = Conv3D(16,(3,2,2),strides=(1,2,2),activation='relu')(x)
x = Flatten()(x)
#x = Dropout(0.5)(x)
x = Dense(32,activation='relu')(x)
x = Dense(16,activation='relu')(x)
outputs = Dense(1,activation=None)(x)
model = Model(inputs=inputs,outputs=outputs)
model.compile(RMSprop(),loss='mean_squared_error')

print(model.summary())

plt.plot(speeds)
plt.plot(model.predict_generator(pred_gen, steps=video_size-sequence_length))
plt.show()

history = model.fit_generator(
	train_gen, 
	steps_per_epoch=int(video_size*split/batch_size), 
	validation_data=val_gen, 
	validation_steps=int(video_size*(1-split)/batch_size),
	epochs=epochs,
	verbose=True)

