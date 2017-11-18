import numpy as np

# Reference:
# http://neuralnetworksanddeeplearning.com/chap2.html

class backPropNN:

	# class instantiation/constructor
	def __init__(self, layerSizes):
		# number of layers in the network
		self.numLayers = len(layerSizes)
		# number of neurons in each layer
		self.layerSizes = layerSizes

		# initialize weights and biases for each layer
		# weights for 1st layer are always 1
		# biases for 1st layer are always 0
		self.weights = []
		self.biases = []
		self.weights = [np.ones([1, layerSizes[0]])]
		self.biases = [np.zeros([layerSizes[0]])]
				
		for i in range(1, len(layerSizes)):
			self.weights.append(np.zeros([layerSizes[i],
				layerSizes[i-1]]))
			self.biases.append(np.zeros([1, layerSizes[i]]))

	# for each l=2..L compute
	# 1. zl=wlal−1+bl
	# 2. al=σ(zl)
	def feedForward(self, x):
		print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside feedForward()')
		z = []
		a = []

		z.append(np.matmul(x, (self.weights[0]).T) + self.biases[0])
		a.append(self.sigmoid(z[0]))

		for i in range(1, self.numLayers):
			z.append(np.matmul(a[i-1], (self.weights[i]).T) \
				+ self.biases[i]) 
			a.append(self.sigmoid(z[i]))
		return [z, a]

	# compute the vector sigma for output layer
	# δL=∇aC⊙σ′(zL)
	def computeErrorInOutputLayerNeurons(self, networkOutput,
		actualOutputs, networkZ):
		print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside computeErrorInOutputLayerNeurons()')
		networkOutput = self.oneHotVectorization(networkOutput)
		dC = networkOutput - actualOutputs
		dSigmoid = self.derivativeOfSigmoid(networkZ)
		outputLayerSigma = np.sum((dC * dSigmoid), axis=0)/\
			networkOutput.shape[0]
		return np.reshape(outputLayerSigma,
			[outputLayerSigma.shape[0], 1])

	# For each l=L−1,L−2,…,2 compute δl=((wl+1)Tδl+1)⊙σ′(zl) 
	def backPropogateError(self, z, a, outputSigma):
		print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside backPropogateError()')
		hiddenLayerSigma = [None] * (self.numLayers - 1)
		hiddenLayerSigma[-1] = outputSigma

		for i in range(len(hiddenLayerSigma) - 2, -1, -1):
			print('shape of weights in the next layer:',
				self.weights[i+2].shape)
			print('shape of hiddenlayersigma in the next layer:',
				hiddenLayerSigma[i+1].shape)
			hiddenLayerSigma[i] = \
				(np.matmul(self.weights[i+2], hiddenLayerSigma[i+1])\
					* self.derivativeOfSigmoid(z[i]))
		return hiddenLayerSigma

	# gradient of cost function
	# ∂C/∂wljk=al−1kδlj and ∂C/∂blj=δlj
	def errorGradientWRTWeights(self, a, sigmas):
		print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside errorGradientWRTWeights()')
		errorGradientWRTWeights = []

		for aLMinusOne, sigmaL in zip(a[:-1], sigmas):
			errorGradientWRTWeights.append(aLMinusOne * sigmaL)
		return errorGradientWRTWeights

	# train the network
	def train(self, x, y, epochs, learningRate, l2Lambda,
		miniBatchSize):
		print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside train()')
		self.stochasticGradientDescent(x, y, epochs, 
			learningRate, l2Lambda, miniBatchSize)

	# run stochastic gradient descent
	def stochasticGradientDescent(self, x, y, epochs, learningRate,
		l2Lambda, miniBatchSize):
		print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside stochasticGradientDescent()')
		N = x.shape[0]

		for i in range(epochs):
			for j in range(int(N/miniBatchSize)):
				lowerBound = j * miniBatchSize
				upperBound = min((j+1)*miniBatchSize, N)

				# step 1 - feedforward - compute error at each layer
				# Input x: Set the corresponding activation 
				# a1 for the input layer.
				# Feedforward: For each l=2,3,…,L
				# compute zl=wlal−1+bl and al=σ(zl).
				[z, a] = self.feedForward(x[lowerBound:upperBound, :])

				# step 2 - compute error in output layer neurons
				# Output error δL: Compute the vector δL=∇aC⊙σ′(zL).
				outputSigma = \
					self.computeErrorInOutputLayerNeurons(a[-1],
						y[lowerBound:upperBound, :], z[-1])

				# step 3 - Backpropagate the error: 
				# For each l=L−1,L−2,…,2 compute δl=((wl+1)Tδl+1)⊙σ′(zl).
				allLayersSigma = self.backPropogateError(z, a
					, outputSigma)

				# step 4 - compute derivative of cost wrt weights
				errorGradientWRTWeights = self.errorGradientWRTWeights(
					a, allLayersSigma)

				# step 5
				self.updateNetworkWeightsAndBiases(errorGradientWRTWeights, 
					allLayersSigma, len(range(lowerBound, upperBound)))

	def updateNetworkWeightsAndBiases(self, errorGradientWRTWeights,
		errorGradientWRTbiases, learningRate, l2Lambda, m):
		print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside updateNetworkWeightsAndBiases()')
		for i in range(len(errorGradientWRTbiases)):
			errorW = (errorGradientWRTWeights[i] + \
				(l2Lambda * self.weights[i+1]))/m
			errorB = (errorGradientWRTbiases[i] + \
				(l2Lambda * self.biases[i+1]))/m

			self.weights[i+1] = self.weights[i+1] - \
				(learningRate * errorW)
			self.biases[i+1] = self.biases[i+1] - \
				(learningRate * errorB) 

	# classify test input
	def classify(self, x):
		print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside classify()')
		[_, a] = self.feedforward(x)
		return self.oneHotVectorization(a[-1])

	# network evaluation
	def evaluateNetwork(self, x, y_):
		y = self.classify(self, x)
		classificationerror = self.classificationError(y, y_)
		print('classification error is:', classificationerror)
		return classificationerror

	# compute classfication error
	def classificationError(self, y, y_):
		print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside classificationError()')
		error = 0

		yClasses = np.argmax(y, axis=1)
		y_Classes = np.argmax(y_, axis=1)

		diff = yClasses - y_Classes

		return 1 - ((np.nonzero(diff == 0))[0].shape[0]/y.shape[0])

	# apply sigmoid to output
	def sigmoid(self, input):
		print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside sigmoid()')
		sigmoidOutput = 1/1 + np.exp(-1 * input)
		return sigmoidOutput

	# derivative of sigmoid function
	def derivativeOfSigmoid(self, input):
		print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside derivativeOfSigmoid()')
		expInput = np.exp(input)
		return expInput/((1+expInput)*(1+expInput))

	# one hot vectorization
	def oneHotVectorization(self, vector):
		print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside oneHotVectorization()')
		for i in range(vector.shape[0]):
			maxProbIndex = np.argmax(vector[i, :])
			vector[i, :] = 0
			vector[i, maxProbIndex] = 1
		return vector

	# describe network, number of layers, number of neurons
	# weight and bias shapes
	def describeNetwork(self):
		print("number of layers in the network:", self.numLayers)
		print("number of neurons in each layer:", self.layerSizes)

		for i in range(len(self.weights)):
			print("shape of weights and biases in layer", str(i), \
				":", self.weights[i].shape, self.biases[i].shape)