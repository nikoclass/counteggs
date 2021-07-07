#!/usr/bin/python3

#Usage for training:
#python3 counteggs.py --trainDir nuevas_imagenes/

#Usage for detection:
#python3 counteggs.py --image all_sin_marron/3.jpg -g false


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
BLUR = 3
ERODE = 0
DILATE = 2
CANNY_MIN = 10
CANNY_MAX = 210


#Fixed parameters (default 300)
IMAGE_WIDTH = 300

X = 0
Y = 1

# import the necessary packages
import imutils
import cv2
import time
import argparse
import math
import numpy as np
import json
import sys
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
ap.add_argument("-v", "--video", required=False, default = "false",
	help="capture from video (default false)")
args = vars(ap.parse_args())


show_graphics = False
if args["graphics"] != "false":
	show_graphics = True

train = False
if args["trainDir"] != "":
	train = True
	show_graphics = False

fromVideo = False
if args["video"] != "false":
	fromVideo = True


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




def getRoundness(countour):
	mean = [0, 0]
	N = len(countour)
	for p in countour:
		mean[X] = mean[X] + p[0][X]
		mean[Y] = mean[Y] + p[0][Y]
	mean[X] = mean[X] / N
	mean[Y] = mean[Y] / N


	distanceMean = 0
	for p in countour:
		distanceMean = distanceMean + math.sqrt(math.pow(mean[X] - p[0][X], 2) + math.pow(mean[Y] - p[0][Y], 2))
	distanceMean = distanceMean / N

	variance = 0
	for p in countour:
		variance = variance + math.fabs(math.sqrt(math.pow(mean[X] - p[0][X], 2) + math.pow(mean[Y] - p[0][Y], 2)) - distanceMean)

	variance = variance / N
	if distanceMean > 0:
		return variance / distanceMean
	else:
		return variance




def saveParametersToFile(filename):
	data = {}
	data['blur'] = BLUR
	data['erode'] = ERODE
	data['dilate'] = DILATE
	data['canny_min'] = CANNY_MIN
	data['canny_max'] = CANNY_MAX
	data['image_width'] = IMAGE_WIDTH
	with open(filename, 'w') as outfile:
		json.dump(data, outfile, indent=4)



class Logger: 
	def __init__(self, filename):
		self.console = sys.stdout
		self.file = open(filename, 'w')
 
	def write(self, message):
		self.console.write(message)
		self.file.write(message)
 
	def flush(self):
		self.console.flush()
		self.file.flush()





if fromVideo == True:

	vs = VideoStream(src=1).start()

	IMAGE_WIDTH *= 2

	while True:
		# read the frame from the camera and send it to the server
		frame = vs.read()
		image = imutils.resize(frame, width=IMAGE_WIDTH)
		
		cnts = detectShape(image)
		eggs = 0

		# loop over the contours
		for c in cnts:
			# draw each contour on the output image with a 3px thick purple
			# outline, then display the output contours one at a time
			if getRoundness(c) > 0.1:
				cv2.drawContours(image, [c], -1, (24, 0, 15), 2)
			else:
				eggs = eggs + 1
				cv2.drawContours(image, [c], -1, (240, 0, 159), 2)
			

		# draw the total number of contours found in purple
		text = "{} eggs".format(eggs)
		cv2.putText(image, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
			(255, 255, 255), 2)


		cv2.imshow("frame", image)
		if cv2.waitKey(1) >= 27:
			exit()

elif train == False:
	# load the input image and show its dimensions, keeping in mind that
	# images are represented as a multi-dimensional NumPy array with
	# shape no. rows (height) x no. columns (width) x no. channels (depth)
	original = cv2.imread(args["image"])

	original = imutils.resize(original, width=IMAGE_WIDTH)


	image = original

	(h, w, d) = image.shape

	if show_graphics:
		cv2.imshow("Original", image)


	cnts = detectShape(image)


	output = original.copy()
	# loop over the contours
	if show_graphics:
		# draw each contour on the output image with a 3px thick purple
		# outline, then display the output contours one at a time
		cv2.drawContours(output, cnts, -1, (240, 0, 159), 2)
		cv2.imshow("Contours", output)


	# draw the total number of contours found in purple
	text = "{} eggs found in file {}".format(len(cnts), args["image"])
	cv2.putText(output, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
		(255, 255, 255), 2)

	if show_graphics:
		cv2.imshow("Contours", output)


	print(text)

else:

	sys.stdout = Logger(args["trainDir"] + "/log.txt")

	print("Reading files...")

	filenames = [f for f in listdir(args["trainDir"]) if isfile(join(args["trainDir"], f))]
	realCount = []
	for filename in filenames:
		if filename != "config.json" and filename != "log.txt":
			print(filename)
			realCount.append(int(filename.strip(".jpgpng")))

	images = []
	for filename in filenames:
		if filename != "config.json" and filename != "log.txt":
			image = cv2.imread(args["trainDir"] + "/" + filename)
			image = imutils.resize(image, width=IMAGE_WIDTH)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			images.append(image)

	print("Training...")

	bestSum = 9999999999999999
	bestBLUR = 0
	bestERODE = 0
	bestDILATE = 0
	bestCANNY_MIN = 0
	best_CANNY_MAX = 0

	for blur in range(1, 23, 2):
		blurImages = []
		for i in range(0, len(images)):
			blurImages.append(cv2.GaussianBlur(images[i], (blur, blur), 0))

		for canny_min in range(10, 150, 10):
			for canny_max in range(canny_min + 10, 250, 10):
				cannyImages = []
				for i in range(0, len(images)):
					cannyImages.append(cv2.Canny(blurImages[i], canny_min, canny_max))


				for dilate in range(0, 6, 1):
					dilateImages = []
					for i in range(0, len(images)):
						dilateImages.append(cv2.dilate(cannyImages[i], None, iterations=dilate))

					for erode in range(0, 6, 1):
						erodeImages = []
						for i in range(0, len(images)):
							erodeImages.append(cv2.erode(dilateImages[i], None, iterations=erode))

						BLUR = blur
						ERODE = erode
						DILATE = dilate
						CANNY_MIN = canny_min
						CANNY_MAX = canny_max
						
						totalError = 0
						totalReal = 0
						sumCount = 0
						diffSum = 0

						for i in range(0,len(images)):
							cnts = cv2.findContours(erodeImages[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
							cnts = len(imutils.grab_contours(cnts))
							
							sumCount = sumCount + cnts
							totalReal = totalReal + realCount[i]
							diff = cnts - realCount[i]
							totalError = totalError + math.fabs(diff)
							diffSum = diffSum + diff * diff 
						
						error = math.floor(100 * (totalError / totalReal))

						if (diffSum < bestSum):
							bestSum = diffSum
							bestBLUR = BLUR
							bestERODE = ERODE
							bestDILATE = DILATE
							bestCANNY_MIN = CANNY_MIN
							best_CANNY_MAX = CANNY_MAX
							sumError = math.floor(100 * math.fabs(totalReal - sumCount) / totalReal)
							print("Best Parameters: BLUR: {}, ERODE: {}, DILATE: {}, CANNY_MIN: {}, CANNY_MAX: {} - diffSum: {} - absolute error: {}% - sum error: {}%"
								.format(blur,erode, dilate, canny_min, canny_max, diffSum, error, sumError))

							saveParametersToFile(args["trainDir"] + "/config.json")
		
						if bestSum == 0:
							exit()



cv2.waitKey(0)
