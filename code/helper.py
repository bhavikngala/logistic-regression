import os
import numpy as np
from scipy import misc

# funtion will read all the images in the given dir
# and flatten them and return the array
def readMNISTData(dataDirectory):
	# initialize empty list
	images = []

	# read all the filenames present in the directory
	filenames = os.listdir(dataDirectory)

	# read each image, flatten it, append to list
	for filename in filenames:
		im = misc.imread(dataDirectory+'/'+filename).flatten('C')
		images.append(im)

	return images