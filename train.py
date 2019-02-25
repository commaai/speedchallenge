import cv2
import matplotlib.pyplot as plt
from generator import data_generator, prediction_generator
from tensorflow.keras.layers import Input, Dense, Flatten, Conv3D, MaxPooling3D, MaxPooling2D, BatchNormalization, Dropout, LeakyReLU, PReLU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD, RMSprop
from opticalflow import denseflow_write
from os.path import splitext
import numpy as np

batch_size = 16
sequence_length = 5
epochs = 20
split = .85
fname, ext = splitext('./data/train.mp4')


print("Processing speeds.")
with open('data/train.txt') as f:
	speeds = f.readlines()
	speeds = np.array([float(x.strip()) for x in speeds]) 

print("Processing video.")
cap = cv2.VideoCapture(fname+ext)
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
	with np.load(fname + '_op.npz') as data:
		video = data['arr_0']
except:
	print("Could not find preprocessed video, creating it now")
	video = denseflow_write(cap, fname)

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
x = Conv3D(32,(1,3,3),strides=(1,2,2),activation=None)(inputs)
x = PReLU()(x)
x = BatchNormalization()(x)
x = Conv3D(32,(1,3,3),strides=(1,2,2),activation=None)(x)
x = PReLU()(x)
x = BatchNormalization()(x)
x = Conv3D(32,(1,3,3),strides=(1,2,2),activation=None)(x) 
x = PReLU()(x)
x = BatchNormalization()(x)

# A convolution across all images together
x = Conv3D(32,(3,1,1),strides=(1,1,1),activation=None)(x)
x = PReLU()(x)
x = BatchNormalization()(x)
x = Conv3D(16,(3,1,1),strides=(1,1,1),activation=None)(x)
x = PReLU()(x)
x = BatchNormalization()(x)
#x = Conv3D(16,(3,2,2),strides=(1,2,2),activation='relu')(x)
x = Flatten()(x)
#x = Dropout(0.5)(x)
x = Dense(32,activation=None)(x)
x = LeakyReLU(alpha=0.1)(x)
x = Dense(16,activation=None)(x)
x = LeakyReLU(alpha=0.1)(x)
outputs = Dense(1,activation=None)(x)
#outputs = LeakyReLU(alpha=1.2)(x)
model = Model(inputs=inputs,outputs=outputs) 
model.compile(RMSprop(),loss='mean_squared_error')

model.summary()

history = model.fit_generator(
	train_gen, 
	steps_per_epoch=int(video_size*split/batch_size), 
	validation_data=val_gen, 
	validation_steps=int(video_size*(1-split)/batch_size),
	epochs=epochs,
	verbose=True,
	callbacks=[ModelCheckpoint('./data/weights.hdf5',save_best_only=True)])

model2 = load_model(filepath='./data/weights.hdf5')
model2.compile(RMSprop(),loss='mean_squared_error')

# Plot the training loss against the validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Loss', 'Validation Loss'])
plt.savefig(fname='./data/lossplot')
#plt.show()

# Plotting predicted speeds against real speeds
plt.plot(model2.predict_generator(pred_gen, steps=video_size-sequence_length))
plt.plot(speeds)
plt.axvline(video_size*split)
plt.xlabel('Frame')
plt.ylabel('Speed in mph')
plt.legend(['Predicted', 'Real'])
plt.savefig(fname='./data/speedplot')
#plt.show()