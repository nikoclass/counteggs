#!/usr/bin/python3
# Changing the following parameters will alter the detection

# These are optimal parameters
BLUR = 9
ERODE = 3
DILATE = 3
CANNY_MIN = 30
CANNY_MAX = 150

THRESHOLD = 210


# Trained parameters
BLUR = 13
ERODE = 0
DILATE = 4
CANNY_MIN = 10
CANNY_MAX = 80




# import the necessary packages
import imutils
import cv2
import time
import argparse
import math
from os import listdir
from os.path import isfile, join


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



if train == False:
	# load the input image and show its dimensions, keeping in mind that
	# images are represented as a multi-dimensional NumPy array with
	# shape no. rows (height) x no. columns (width) x no. channels (depth)
	original = cv2.imread(args["image"])

	original = imutils.resize(original, width=300)

	image = original

	(h, w, d) = image.shape

	if show_graphics:
		cv2.imshow("Original", image)


	cnts = detectShape(image)

	output = original.copy()
	# loop over the contours
	if show_graphics:
		for c in cnts:
			# draw each contour on the output image with a 3px thick purple
			# outline, then display the output contours one at a time
			cv2.drawContours(output, [c], -1, (240, 0, 159), 2)
			cv2.imshow("Contours", output)


	# draw the total number of contours found in purple
	text = "{} eggs found in file {}".format(len(cnts), args["image"])
	cv2.putText(output, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
		(255, 255, 255), 2)

	if show_graphics:
		cv2.imshow("Contours", output)


	print(text)

else:
	print("training...")

	filenames = [f for f in listdir(args["trainDir"]) if isfile(join(args["trainDir"], f))]
	realCount = []
	for filename in filenames:
		realCount.append(int(filename.strip(".jpgpng")))

	images = []
	for filename in filenames:
		image = cv2.imread(args["trainDir"] + "/" + filename)
		image = imutils.resize(image, width=300)
		images.append(image)


	bestSum = 9999999999999999
	bestBLUR = 0
	bestERODE = 0
	bestDILATE = 0
	bestCANNY_MIN = 0
	best_CANNY_MAX = 0

	for blur in range(1, 23, 2):
		for erode in range(0, 9, 1):
			for dilate in range(erode, 9, 1):
				for canny_min in range(10, 150, 10):
					for canny_max in range(canny_min + 10, 250, 10):

						BLUR = blur
						ERODE = erode
						DILATE = dilate
						CANNY_MIN = canny_min
						CANNY_MAX = canny_max
						
						totalError = 0
						totalReal = 0
						diffSum = 0

						for i in range(0,len(images)):
							cnts = detectShape(images[i])
							totalReal = totalReal + realCount[i]
							diff = len(cnts) - realCount[i]
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
							print("Best Parameters: BLUR: {}, ERODE: {}, DILATE: {}, CANNY_MIN: {}, CANNY_MAX: {} - diffSum: {} - absolute error: {}%".format(blur,erode, dilate, canny_min, canny_max, diffSum, error))
		
						if bestSum == 0:
							exit()



cv2.waitKey(0)
