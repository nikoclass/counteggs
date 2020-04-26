#!/usr/bin/python3
# Changing the following parameters will alter the detection

# These are optimal parameters
#BLUR = 9
#ERODE = 3
#DILATE = 3
#CANNY_MIN = 30
#CANNY_MAX = 150

#not used parameter
#THRESHOLD = 210


# Trained parameters
BLUR = 7
ERODE = 5
DILATE = 5
CANNY_MIN = 100
CANNY_MAX = 120


#Fixed parameters (default 300)
IMAGE_WIDTH = 600

# import the necessary packages
import imutils
import cv2
import time
import argparse
import math
import numpy as np
from os import listdir
from os.path import isfile, join

from imutils.video import VideoStream



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,	help="path to input image")
ap.add_argument("-g", "--graphics", required=False, default = "false",
	help="show graphics (default false)")
ap.add_argument("-t", "--trainDir", required=False, default = "",
	help="trainDir (default empty)")
args = vars(ap.parse_args())


show_graphics = False
if args["graphics"] != "false":
	show_graphics = True

train = False
if args["trainDir"] != "":
	train = True
	show_graphics = False






def detectShape(image):
	# convert the image to grayscale
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# apply a Gaussian blur with a 11x11 kernel to the image to smooth it,
	# useful when reducing high frequency noise
	image = cv2.GaussianBlur(image, (BLUR, BLUR), 0)
	if show_graphics:
		cv2.imshow("Blur", image)


	# threshold the image by setting all pixel values less than 225
	# to 255 (white; foreground) and all pixel values >= 225 to 255
	# (black; background), thereby segmenting the image

	#image = cv2.threshold(image, THRESHOLD, 255, cv2.THRESH_BINARY)[1]
	#image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,5,2)

	# we use Canny edje detection instead of thresholding because 
	# it provides robust detection regardless of lightning conditions
	image = cv2.Canny(image, CANNY_MIN, CANNY_MAX)
	
	
	if show_graphics:
		cv2.imshow("Thresh", image)


	# dilate then erode to provide better clustering
	image = cv2.dilate(image, None, iterations=DILATE)
	image = cv2.erode(image, None, iterations=ERODE)

	if show_graphics:
		cv2.imshow("Eroded", image)


	# find contours (i.e., outlines) of the foreground objects in the
	# thresholded image
	cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	return cnts






vs = VideoStream(src=0).start()

while True:
	# read the frame from the camera and send it to the server
	frame = vs.read()
	image = imutils.resize(frame, width=IMAGE_WIDTH)
	
	cnts = detectShape(image)
	# loop over the contours
	for c in cnts:
		# draw each contour on the output image with a 3px thick purple
		# outline, then display the output contours one at a time
		cv2.drawContours(image, [c], -1, (240, 0, 159), 2)
		

	# draw the total number of contours found in purple
	text = "{} eggs".format(len(cnts))
	cv2.putText(image, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
		(255, 255, 255), 2)


	cv2.imshow("frame", image)
	if cv2.waitKey(1) >= 27:
		exit()

