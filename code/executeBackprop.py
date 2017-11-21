import numpy as np
# import image_helper
from helpers import  image_helper as davinci

# import MNIST data module
from tensorflow.examples.tutorials.mnist import input_data

import backPropNN as backPropNetwork

def main():
	# constants
	epochs = 22
	learningRate = 0.5
	l2Lambda = 1
	miniBatchSize = 100

	# initialize network
	backpropnetwork = backPropNetwork.backPropNN([784, 100, 10])

	# describe network
	backpropnetwork.describeNetwork()

	# load MNIST data
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	# train network
	backpropnetwork.train(mnist.train.images, mnist.train.labels,
		epochs, learningRate, l2Lambda, miniBatchSize, mnist.test.images,
		mnist.test.labels)

	# evaluate network on test data
	backpropnetwork.evaluateNetwork(mnist.test.images, mnist.test.labels)

	# # save network
	# # TODO: fix errors
	# backpropnetwork.saveNetwork('./../results/backpropNN_take00/')

if __name__ == '__main__':
	main()