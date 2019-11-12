import numpy as np
import cv2
import time
import os
import matplotlib.pyplot as plt

train = "data\\train.mp4"
train_speed = "data\\train.txt"
test = "data\\test.mp4"

# create images directory
if not os.path.exists('images'):
    os.makedirs('images')

def read_file(train):
    cap = cv2.VideoCapture(train)
    ret, frame = cap.read()
    return ret, frame

def crop_frame(frame):
    """Crops frame to remove sky and dashboard
    input: frame (RGB)
    returns: croped frame
    """
    crop_frame = frame[50:-170,:]
    return crop_frame

def hist_eq(frame):
    """Converts RGB frame to YCrCb and performs histogram equalization
    on brightness channel.
    input: frame (RGB)
    returns: frame with equalized brightness
    """
    # convert to YCbCr
    ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(ycrcb_frame)

    # histogram equalization on brightness channel 
    equ = cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb_frame)
    
    # convert to RGB
    rgb_frame = cv2.cvtColor(ycrcb_frame, cv2.COLOR_YCrCb2BGR) 
    return rgb_frame

def resize_frame(frame, width, height): 
    """Resize frame
    input: frame (RGB)
    returns: resized frame
    """
    # set width = 220 and height = 66
    return cv2.resize(frame, (width, height))

def save_frame(label, frame):
    """Saves frame as an image.
    input: frame number, frame (RGB)
    """
    name = './images/frame_' + str(index) + '.jpg'
    print ('Creating...' + name)
    cv2.imwrite(name, frame)

def prepare_trainset(frames):
    pass

def prepare_testset(frames):
    pass

def segmenatation_and_del(frame):
    """Perform semantic segmentation and remove unnecessary elements.
    input: frame (RGB)
    return: processed frame
    """
    pass

index = 0
cap = cv2.VideoCapture(train)

while(cap.isOpened()): 

    ret, frame = cap.read()
    frame = crop_frame(frame)
    hist_frame = hist_eq(frame)
    resized_frame = resize_frame(hist_frame, 220, 66)
    # save_frame(index, resized_frame)
    cv2.imshow('resized', hist_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

    index += 1
