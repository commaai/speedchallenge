import cv2
from generator import data_generator
from tensorflow.keras.layers import Input, Dense, Flatten, Conv3D, MaxPooling3D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

batch_size = 128
sequence_length = 3
epochs = 200

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

gen = data_generator(cap, video_size, batch_size, sequence_length, (width,height), speeds='train.txt')

# Will return a feature and label set.
# Features are a list of image sequences in the form: (sequence_length, img_height, img_width, dimensions)
inputs = Input((sequence_length,height,width,3))

# A convolution being applied to each image seperately
x = Conv3D(32,(1,5,5),strides=(1,3,3),activation='relu')(inputs) 
x = Conv3D(32,(1,3,3),strides=(1,2,2),activation='relu')(x)
x = Conv3D(32,(1,2,2),strides=(1,2,2),activation='relu')(x)

# A convolution across all images together
x = Conv3D(32,(sequence_length,5,5),strides=(1,2,2),activation='relu')(x)

x = MaxPooling3D((1,2,2))(x) #Pooling this convolution
x = Flatten()(x) 
x = Dense(128,activation='relu')(x)
x = Dense(32,activation='relu')(x)
outputs = Dense(1,activation='relu')(x)
model = Model(inputs=inputs,outputs=outputs)
model.compile(SGD(0.01,momentum=0.9),loss='mean_squared_error')

print(model.summary())

history = model.fit_generator(gen,steps_per_epoch=int(video_size/batch_size),epochs=epochs,verbose=True)

