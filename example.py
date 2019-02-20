import cv2
cap = cv2.VideoCapture('data/train.mp4')

if (cap.isOpened()== False): 
  print("Error opening video stream or file")
  exit()

# Read until video is completed
while(cap.isOpened()):
	# Capture frame-by-frame
	ret, frame = cap.read()
	if ret == True:
		# Display the resulting frame
		cv2.imshow('Frame',frame)

		# Press Q on keyboard to  exit
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break
	# Break the loop
	else: 
		break

# When everything done, release the video capture object
cap.release()
