# https://www.oreilly.com/learning/not-another-mnist-tutorial-with-tensorflow
# https://www.tensorflow.org/get_started/mnist/pros

# import MNIST data module
from tensorflow.examples.tutorials.mnist import input_data

# load MNIST data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

# create interaci=tive session
# session = tf.InteractiveSession()

# we build ccomputation graphs before starting the sessions
# sessions are connection to the backend that run the graphs
# interactive session allows to be flexible, build stuff on the go
# otherwise entire graph has to be pre built before launching session

# tensor of any number of rows and 784 columns
x = tf.placeholder(tf.float32, [None, 784])

# vairables can be  modified by computations
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# softmax model 
y = tf.nn.softmax(tf.matmul(x, W) + b)