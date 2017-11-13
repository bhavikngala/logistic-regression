from scipy import misc
import matplotlib.pyplot as plt

def readImage(filename):
	return misc.imread(filaname)

def resizeImage(img, outputSize, interpMethod):
	return misc.imresize(img, outputSize, interp = interpMethod)

def showImage(img):
	plt.imshow(img)
	plt.show()