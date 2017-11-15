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

def generateImageTensorsAndLbls(directory, numbers):
	images = []
	lbls = []
	for number in numbers:
		imgs = batchReadAndResizeImages(directory+'/'+number,
			[28 28], 'bilinear')
		lbl = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		lbl[number] = 1
		lbl = [lbl] * imgs.shape[0]

		images = images + imgs
		lbls = lbls + lbl