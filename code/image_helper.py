from scipy import misc
import matplotlib.pyplot as plt
import os

def readImage(filename):
	return misc.imread(filaname)

def resizeImage(img, outputSize, interpMethod):
	return misc.imresize(img, outputSize, interp = interpMethod)

def showImage(img):
	plt.imshow(img)
	plt.show()

def batchReadAndResizeImages(directory, outputSize, interpMethod):
	# 2D array of images
	imgs = np.array()
	return imgs