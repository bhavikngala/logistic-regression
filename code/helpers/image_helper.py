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
			img = normalizeImage(img)
			img = img.flatten()
			imgs.append(img)
	return imgs

def normalizeImage(img):
	return img/255

def readUSPSTestImagesAndLbls(directory):
	images = []
	lbls = []
	
	images = batchReadAndResizeImages(directory, [28, 28], 'bilinear')

	for i in range(0, 10):
		lbl = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		lbl[i] = 1
		lbl = [lbl] * 150
		lbls = lbl + lbls

	return [images, lbls]