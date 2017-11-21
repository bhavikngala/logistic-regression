import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from helpers import  image_helper as davinci

def main():
	mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

	x = tf.placeholder(tf.float32, [None, 784])
	y_true = tf.placeholder(tf.float32,[None, 10])

	W1 = tf.Variable(tf.truncated_normal([784, 100],stddev=0.1))
	b1 = tf.Variable(tf.truncated_normal([100], stddev=0.1))

	activation1 = tf.nn.relu(tf.matmul(x, W1) + b1)

	W2 = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1))
	b2 = tf.Variable(tf.truncated_normal([10], stddev=0.1))

	z2 = tf.matmul(activation1, W2) + b2

	cross_entropy = \
		tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(
				labels=y_true, logits=z2))

	train_step = \
		tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	# test the model
	correct_prediction = \
		tf.equal(tf.argmax(z2, 1), tf.argmax(y_true, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	# Training
	for i in range(1000):
	    batch_xs, batch_ys = mnist.train.next_batch(128)
	    l,_,a = sess.run([cross_entropy, train_step, accuracy], 
	                     feed_dict={x: batch_xs, y_true: batch_ys})
	    if i%100 == 0 or (i<100 and i%10==0):
	        print(str(i)+': loss: '+str(l)+' accuracy: '+str(a))

	print('accuracy on MNIST test data:', sess.run(accuracy,
		feed_dict={x: mnist.test.images, y_true: mnist.test.labels}))

	directory = './../data/Numerals'
	[uspsNumeralsImgs, uspsNumeralsLbls] = \
		davinci.readUSPSTrainImagesAndLbls(directory)
	print('accuracy on USPS Numerals folder data:', sess.run(accuracy,
		feed_dict={x: uspsNumeralsImgs, y_true: uspsNumeralsLbls}))

	directory = './../data/Test'
	[uspsTestImgs, uspsTestLabels] = \
		davinci.readUSPSTestImagesAndLbls(directory)
	print('accuracy on USPS test folder data:', sess.run(accuracy,
		feed_dict={x: uspsTestImgs, y_true: uspsTestLabels}))

if __name__ == '__main__':
	main()