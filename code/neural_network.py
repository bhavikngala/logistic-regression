# https://www.oreilly.com/learning/not-another-mnist-tutorial-with-tensorflow
# https://www.tensorflow.org/get_started/mnist/pros

import numpy as np
# import image_helper
from helpers import  image_helper as davinci

# import MNIST data module
from tensorflow.examples.tutorials.mnist import input_data

# load MNIST data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

# tensor of any number of rows and 784 columns
x = tf.placeholder(tf.float32, [None, 784])

# vairables can be  modified by computations
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# softmax model, predicted labels 
y = tf.nn.softmax(tf.matmul(x, W) + b)
# correct label
y_ = tf.placeholder(tf.float32, [None, 10])

# cross entropy function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
	reduction_indices=[1]))

# train backprop with gradient descent
train_step = tf.train.GradientDescentOptimizer(0.5)\
	.minimize(cross_entropy)

# create interaci=tive session
sess = tf.InteractiveSession()

# we build ccomputation graphs before starting the sessions
# sessions are connection to the backend that run the graphs
# interactive session allows to be flexible, build stuff on the go
# otherwise entire graph has to be pre built before launching session

# operation to initialize the variables we created
tf.global_variables_initializer().run()

# read usps images
directory = './../data/Numerals'
[uspsTrainImgs, uspsTrainLbls] = \
	davinci.readUSPSTrainImagesAndLbls(directory)

# convert to np array
uspsTrainImgs = np.array(uspsTrainImgs)
uspsTrainLbls = np.array(uspsTrainLbls)

# concatenate data with mnist data
mnist_usps_train_imgs = np.concatenate([mnist.train.images,
	uspsTrainImgs])
mnist_usps_train_lbls = np.concatenate([mnist.train.labels,
	uspsTrainLbls]) 
N, _ = mnist_usps_train_imgs.shape

# run training step 1000 times
'''
for _ in range(1000):
	# batch of random 100 images
	batch_xs, batch_ys = mnist.train.next_batch(100)
	# feed the batch in placeholder variables
	# this is stochastic training
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
'''
for _ in range(1000):
	for i in range(int(N/100)):
		lowerBound = i * 100
		upperBound = min((i+1)*100, N)
		sess.run(train_step, feed_dict={
			x: mnist_usps_train_imgs[lowerBound:upperBound, :],
			y_:mnist_usps_train_lbls[lowerBound:upperBound, :]})

# model evaluation
# model to evaluate accuracy
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# run the network model on test images and run the accuracy model
# on output for test images
print(sess.run(accuracy,
	feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# read the USPS data
directory = './../data/Test'
[images, lbls] = davinci.readUSPSTestImagesAndLbls(directory)

images = np.array(images)
lbls = np.array(lbls)

# run model on USPS images and pring accuracy
print(sess.run(accuracy,
	feed_dict={x: images, y_: lbls}))

davinci.plotImage(images[0, :])
davinci.plotImage(mnist.train.images[0, :])