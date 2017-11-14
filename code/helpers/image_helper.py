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
	imgs = []
	for file in os.listdir(directory):
		if file.endswith('.png'):
			img = misc.imread(directory+'/'+file, flatten=True);
			img = resizeImage(img, outputSize, interpMethod)
			img = img.flatten()
			imgs.append(img)
	return imgs