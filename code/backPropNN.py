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
		#self.weights = [np.ones([layerSizes[0], 1])]
		#self.biases = [np.zeros([layerSizes[0], 1])]
				
		for i in range(1, len(layerSizes)):
			self.weights.append(np.zeros([layerSizes[i],
				layerSizes[i-1]]))
			self.biases.append(np.zeros([layerSize[i], 1]))

	# for each l=2..L compute
	# 1. zl=wlal−1+bl
	# 2. al=σ(zl)
	def feedForward(self, x):
		z = []
		a = []

		z.append(np.matmul(x, (self.weights[0]).T) + self.biases[0])
		a.append(self.sigmoid(z[0]))

		for i in range(1, self.numLayers - 1):
			z.append(np.matmul(a[i-1], (self.weights[i]).T) \
				+ self.biases[i]) 
			a.append(sigmoid(self, z))
		return [z, a]

	# compute the vector sigma for output layer
	# δL=∇aC⊙σ′(zL)
	def computeErrorInOutputLayerNeurons(self, networkOutput,
		actualOutputs, networkZ):
		networkOutput = self.oneHotVectorization(self, networkOutput)
		dC = networkOutput - actualOutputs
		dSigmoid = self.derivativeOfSigmoid(net)
		return dC * derivativeOfSigmoid(self, networkZ)

	# For each l=L−1,L−2,…,2 compute δl=((wl+1)Tδl+1)⊙σ′(zl) 
	def backPropogateError(self, z, a, outputSigma):
		hiddenLayerSigma = [None] * (len(self.numLayers) - 1)
		hiddenLayerSigma[-1] = outputSigma

		for i in range(len(hiddenLayerSigma) - 2, -1, -1):
			hiddenLayerSigma[i] = \
				(np.matmul(self.weights[i+1], hiddenLayerSigma) *
				self.derivativeOfSigmoid(z[i]))
		return hiddenLayerSigma

	# gradient of cost function
	# ∂C/∂wljk=al−1kδlj and ∂C/∂blj=δlj
	def errorGradientWRTWeights(self, a, sigmas):
		errorGradientWRTWeights = []

		for aLMinusOne, sigmaL in zip(a[:-1], sigmas):
			errorGradientWRTWeights.append(aLMinusOne * sigmaL)
		return errorGradientWRTWeights

	# train the network
	def train(self, x, y, epochs, learningRate, l2Lambda,
		miniBatchSize):
		self.stochasticGradientDescent(self, x, y, epochs, 
			learningRate, l2Lambda, miniBatchSize)
		return None

	# run stochastic gradient descent
	def stochasticGradientDescent(self, x, y, epochs, learningRate,
		l2Lambda, miniBatchSize):
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
				[z, a] = self.feedforward(self,
					x[lowerBound:upperBound, :])

				# step 2 - compute error in output layer neurons
				# Output error δL: Compute the vector δL=∇aC⊙σ′(zL).
				outputSigma = \
					self.computeErrorInOutputLayerNeurons(self,
					a[-1], y[lowerBound:upperBound, :],
					networkZ[-1])

				# step 3 - Backpropagate the error: 
				# For each l=L−1,L−2,…,2 compute δl=((wl+1)Tδl+1)⊙σ′(zl).
				hiddenLayerSigma = self.backPropogateError(self, z, a
					outputSigma)

				# step 4 - compute derivative of cost wrt weights
				errorGradientWRTWeights = self.errorGradientWRTWeights(
					self, a, hiddenLayerSigma + outputSigma)

				# step 5
				self.updateWeightsAndBiases(self,
					errorGradientWRTWeights, 
					hiddenLayerSigma + outputSigma)

	# classify test input
	def classify(self):
		return None

	# network evaluation
	def evaluateNetwork(self):
		return None

	# compute classfication error
	def classificationError(self):
		return None


	# apply sigmoid to output
	def sigmoid(self, input):
		input = 1 + np.exp(-1 * input)
		return 1/input

	# derivative of sigmoid function
	def derivativeOfSigmoid(self, input):
		expInput = np.exp(input)
		return expInput/((1+expInput)*(1+expInput))

	# one hot vectorization
	def oneHotVectorization(self, vector):
		maxProbIndex = np.argmax(vector)
		vector[:] = 0
		vector[maxProbIndex] = 1
		return vector
