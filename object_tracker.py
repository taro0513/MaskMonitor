# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from tracker.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
# vs = VideoStream(src='pan.mp4').start()
vs = cv2.VideoCapture('test.mp4')
time.sleep(2.0)
save = []
# loop over the frames from the video stream
while True:
	# read the next frame from the video stream and resize it
	_, frame = vs.read()
	# frame = imutils.resize(frame, width=400)

	# if the frame dimensions are None, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the frame, pass it through the network,
	# obtain our output predictions, and initialize the list of
	# bounding box rectangles
	blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
		(104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()
	# print(detections.shape[2])
	rects = []
	countt = 0
	for i in range(0, detections.shape[2]):
		if detections[0, 0, i, 2] > args["confidence"]:
			countt += 1
	print(countt)

	for i in range(0, detections.shape[2]):

		if detections[0, 0, i, 2] > args["confidence"]:
			print(f"confidence:　{detections[0, 0, i, 2]}")

			box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
			print(f"box: {box}")
			rects.append(box.astype("int"))

			(startX, startY, endX, endY) = box.astype("int")
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 255, 0), 2)

	objects = ct.update(rects)

	for (objectID, centroid) in objects.items():

		text = "ID {}".format(objectID)
		if objectID not in save:
			cv2.imwrite(f'O-{objectID}.jpg', frame)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break
	# break
cv2.destroyAllWindows()
vs.stop()