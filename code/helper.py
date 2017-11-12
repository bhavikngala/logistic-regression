from tensorflow.examples.tutorials.mnist import input_data

def readMNISTData():
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	return [mnist.train.images, mnist.train.labels,
			mnist.validation.images, mnist.validation.labels,
			mnist.test.images, mnist.test.labels]