import matplotlib.pyplot as plt
from generator import data_generator, prediction_generator
from tensorflow.keras.layers import Input, Dense, Flatten, Conv3D, MaxPooling3D, MaxPooling2D, BatchNormalization, Dropout, LeakyReLU, PReLU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD, RMSprop
from opticalflow import denseflow
from os.path import splitext
import numpy as np

batch_size = 16
sequence_length = 8
epochs = 25
split = .9

print("Processing speeds.")
with open('data/train.txt') as f:
	speeds = f.readlines()
	speeds = np.array([float(x.strip()) for x in speeds])
	speeds = speeds

print("Processing video.")
fname = "data/train.mp4"
try:
	f, ext = splitext(fname)
	with np.load(f + '_op.npz') as data:
		video = data['arr_0']
except:
	print("Could not find preprocessed video, creating it now")
	video = denseflow(fname, 4)

width = video.shape[2]
height = video.shape[1]
video_size = len(video)

train_gen = data_generator(video[:int(video_size*split)], speeds[:int(video_size*split)], batch_size, sequence_length)
val_gen = data_generator(video[int(video_size*split):], speeds[int(video_size*split):], batch_size, sequence_length)
pred_gen = prediction_generator(video, sequence_length)

# Will return a feature and label set.	
# Features are a list of image sequences in the form: (sequence_length, img_height, img_width, dimensions)
inputs = Input((sequence_length,height,width,3))

# A convolution being applied to each image seperately
x = Conv3D(32,(1,3,3),strides=(1,2,2),activation=None)(inputs)
x = LeakyReLU(alpha=0.1)(x)
x = BatchNormalization()(x)
x = Conv3D(32,(3,3,3),strides=(2,2,2),activation=None)(x)
x = LeakyReLU(alpha=0.1)(x)
x = BatchNormalization()(x)
x = Conv3D(32,(3,3,3),strides=(2,2,2),activation=None)(x)
x = LeakyReLU(alpha=0.1)(x)
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dropout(.5)(x)

x = Dense(32,activation=None)(x)
x = LeakyReLU(alpha=0.1)(x)
x = Dense(16,activation=None)(x)
x = LeakyReLU(alpha=0.1)(x)
outputs = Dense(1,activation=None)(x)
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

plt.clf()

# Plotting predicted speeds against real speeds
plt.plot(model2.predict_generator(pred_gen, steps=video_size-sequence_length))
plt.plot(speeds)
plt.xlabel('Frame')
plt.ylabel('Speed in mph')
plt.legend(['Predicted', 'Real'])
plt.savefig(fname='./data/speedplot')
#plt.show()

