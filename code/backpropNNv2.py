import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from helpers import fileHelper as ironman

class BackPropNNv2:

	def __init__(self, layerSizes):
		self.numLayers = len(layerSizes)
		self.layerSizes = layerSizes
		self.weights = []
		self.biases = []

		for i in range(1, len(layerSizes)):
			self.weights.append(np.random.rand(layerSizes[i],
				layerSizes[i-1]))
			self.biases.append(np.random.rand(layerSizes[i], 1))

	def feedForward(self, x):
		z = [x.T]
		a = [z[0]]

		for i in range(0, len(self.weights)):
			z.append(np.matmul(self.weights[i], a[i]) + self.biases[i])
			a.append(self.sigmoid(z[i+1]))
		return [z, a]

	def backPropogateErrors(self, x_row, y_row):
		deltas = []
		errorInWeights = []

		for i in range(len(self.weights)):
			deltas.append(np.zeros(self.biases[i].shape))
			errorInWeights.append(np.zeros(self.weights[i].shape))
		
		[z, a] = self.feedForward(np.reshape(x_row, [1, x_row.shape[0]]))

		delta = (a[-1] - np.reshape(y_row, [y_row.shape[0], 1])) * \
			self.derivativeOfSigmoid(z[-1])
		deltas[-1] = delta

		errorInWeights[-1] = np.dot(delta, a[-2].T)

		for i in range(2, len(deltas)+1):
			delta = np.dot(self.weights[-i+1].T, delta) * \
				self.derivativeOfSigmoid(z[-i])
			deltas[-i] = delta

			errorInWeights[-i] = np.dot(delta, a[-i-1].T)

		return [deltas, errorInWeights]

	def computeWeightsAndBiases(self, x, y):
		errorInBiases = []
		errorInWeights = []

		for i in range(len(self.weights)):
			errorInBiases.append(np.zeros(self.biases[i].shape))
			errorInWeights.append(np.zeros(self.weights[i].shape))

		for x_row, y_row in zip(x, y):
			errorInBiasesRow, errorInWeightsRow = self.backPropogateErrors(
				x_row, y_row)
			for i in range(len(errorInBiases)):
				errorInBiases[i] += errorInBiasesRow[i]
				errorInWeights[i] += errorInWeightsRow[i]

		return [errorInBiases, errorInWeights]


	def updateWeightsAndBiases(self, errorInWeights, errorInBiases,
		learningRate, m):
		for i in range(len(self.weights)):
			self.weights[i] -= (learningRate/m) * errorInWeights[i]
			self.biases[i] -= (learningRate/m) * errorInBiases[i]

	def trainNetwork(self, train_x, train_y, epochs, learningRate,
		miniBatchSize, vali_x=None, vali_y=None):
		N = train_x.shape[0]

		for i in range(epochs):
			print('epoch:', str(i))
			for j in range(int(N/miniBatchSize)):
				lowerBound = i * miniBatchSize
				upperBound = min((i+1) * miniBatchSize, N)

				[errorInBiases, errorInWeights] = \
					self.computeWeightsAndBiases(
						train_x[lowerBound:upperBound],
						train_y[lowerBound:upperBound, :])

				self.updateWeightsAndBiases(errorInWeights, errorInBiases,
					learningRate, len(range(lowerBound, upperBound)))

			if (vali_x is not None and vali_y is not None) and i % 5 == 0:
				self.evaluateNetwork(vali_x, vali_y)

	def evaluateNetwork(self, x, y):
		[_, a] = self.feedForward(x)
		Nwrong = 0

		for a_row, y_row in zip(a[-1].T, y):
			if np.argmax(a_row) != np.argmax(y_row):
				Nwrong += 1

		print('classification error:', Nwrong/y.shape[0])

	def sigmoid(self, z):
		return 1.0/(1.0 + np.exp(-z))

	def derivativeOfSigmoid(self, z):
		return self.sigmoid(z) * (1 - self.sigmoid(z))

	# describe networks
	def describeNetwork(self):
		print("number of layers in the network:", self.numLayers)
		print("number of neurons in each layer:", self.layerSizes)

		for i in range(len(self.weights)):
			print("shape of weights and biases in layer", str(i), \
				":", self.weights[i].shape, self.biases[i].shape)

	def describeArrayShapesInList(self, list, msgString):
		print('describing list:', msgString)
		for l in list:
			print(msgString, 'shape:', l.shape)

def main():
	epochs = 15
	learningRate = 0.5
	miniBatchSize = 100

	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	nn = BackPropNNv2([784, 50, 10])
	nn.describeNetwork()
	nn.trainNetwork(mnist.train.images, mnist.train.labels, 
		epochs, learningRate, miniBatchSize,
		mnist.validation.images, mnist.validation.labels)

	nn.evaluateNetwork(mnist.test.images, mnist.test.labels)

if __name__ == '__main__':
	main()