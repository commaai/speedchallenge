import cv2 as cv
import numpy as np
import time
from os.path import splitext

def denseflow(fname, resize_factor):
    cap = cv.VideoCapture(fname)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
        exit()

    # Get the video dimensions, 3 is the ordinal value of CV_CAP_PROP_FRAME_WIDTH, 4 is CV_CAP_PROP_FRAME_HEIGHT
    # Also resize them because these images are too big
    width = int(cap.get(3) / resize_factor)
    height = int(cap.get(4) / resize_factor)
    shape = (width,height)
    size = int(cap.get(7))
    ret, frame1 = cap.read()
    frame1 = cv.resize(frame1,shape)
    prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255

    frames = np.zeros((size,hsv.shape[0],hsv.shape[1],3))
    frames[0] = hsv

    fcount = 1
    t = time.time()

    while(1):
        ret, frame2 = cap.read()
        if not ret:
            break
        frame2 = cv.resize(frame2,shape)
        next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
        frames[fcount] = hsv
        prvs = next
        fcount += 1
    cap.release()

    f, ext = splitext(fname)
    np.savez(f+'_op', frames)
    return frames

def lkflow():
    cap = cv.VideoCapture('data/train.mp4')
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    fcount = 0
    t = time.time()
    while(1):

        if fcount != 0 and fcount % 20 == 0:
            print("Avg fps: ", 20 / (time.time() - t))
            t = time.time()
        ret,frame = cap.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv.add(frame,mask)
        cv.imshow('frame',img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
        fcount += 1
    cv.destroyAllWindows()
    cap.release()

def denseflow_show():
    cap = cv.VideoCapture("data/train.mp4")
    ret, frame1 = cap.read()
    prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    cap.set(1,15000)
    while(1):
        ret, frame2 = cap.read()
        if not ret:
            break
        next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
        cv.imshow('frame2',bgr)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv.imwrite('opticalfb.png',frame2)
            cv.imwrite('opticalhsv.png',bgr)
        prvs = next
    cap.release()
    cv.destroyAllWindows()