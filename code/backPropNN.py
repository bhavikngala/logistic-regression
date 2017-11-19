import numpy as np
from helpers import fileHelper as ironman

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
		self.biases = [np.zeros([1, layerSizes[0]])]
				
		for i in range(1, len(layerSizes)):
			self.weights.append(np.random.rand(layerSizes[i-1],
				layerSizes[i]))
			self.biases.append(np.random.rand(1, layerSizes[i]))

	# for each l=2..L compute
	# 1. zl=wlal−1+bl
	# 2. al=σ(zl)
	def feedForward(self, x):
		#print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside feedForward()')
		z = []
		a = []

		z.append(x)
		# a.append(self.sigmoid(z[0]))
		# why did i think i had to take sigmoid here???
		a.append(z[0])

		for i in range(1, self.numLayers):
			z.append(np.matmul(a[i-1], self.weights[i]) \
				+ self.biases[i]) 
			a.append(self.sigmoid(z[i]))

		return [z, a]

	# compute the vector sigma for output layer
	# δL=∇aC⊙σ′(zL)
	def computeErrorInOutputLayerNeurons(self, networkOutput,
		actualOutputs, networkZ):
		#print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside computeErrorInOutputLayerNeurons()')
		dC = networkOutput - actualOutputs
		dSigmoid = dC * self.derivativeOfSigmoid(networkZ)
		return dSigmoid

	# For each l=L−1,L−2,…,2 compute δl=((wl+1)Tδl+1)⊙σ′(zl) 
	def backPropogateError(self, z, a, outputSigma):
		#print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside backPropogateError()')
		hiddenLayerSigma = [None] * (self.numLayers - 1)
		hiddenLayerSigma[-1] = outputSigma

		for i in range(len(hiddenLayerSigma) - 2, -1, -1):
			hiddenLayerSigma[i] = \
				(np.matmul(hiddenLayerSigma[i+1], self.weights[i+2].T)\
					* self.derivativeOfSigmoid(z[i+1]))

		return hiddenLayerSigma

	# gradient of cost function
	# ∂C/∂wljk=al−1kδlj and ∂C/∂blj=δlj
	def errorGradientWRTWeights(self, a, sigmas):
		#print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside errorGradientWRTWeights()')
		errorGradientWRTWeights = []

		for aLMinusOne, sigmaL in zip(a[:-1], sigmas):
			'''
			activations for each input will be multiplied by
			neuron errors for that corresponding neurons
			take average for all the w*sigma for inputs in batch
			refer the photo of the writings on green board at home
			'''
			errorGradientWRTWeightsInCurrentLayer = \
				np.zeros([aLMinusOne.shape[1], sigmaL.shape[1]])

			for a,s in zip(aLMinusOne, sigmaL):
				a = np.reshape(a, [a.shape[0], 1])

				errorGradientWRTWeightsInCurrentLayer += s * a

			errorGradientWRTWeightsInCurrentLayer /= sigmaL.shape[0]

			errorGradientWRTWeights.append(
				errorGradientWRTWeightsInCurrentLayer)
		return errorGradientWRTWeights

	# train the network
	def train(self, x, y, epochs, learningRate, l2Lambda,
		miniBatchSize):
		#print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside train()')
		self.stochasticGradientDescent(x, y, epochs, 
			learningRate, l2Lambda, miniBatchSize)

	# run stochastic gradient descent
	def stochasticGradientDescent(self, x, y, epochs, learningRate,
		l2Lambda, miniBatchSize):
		#print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside stochasticGradientDescent()')
		N = x.shape[0]

		for i in range(epochs):
			print('epoch:', str(i))
			for j in range(int(N/miniBatchSize)):
				lowerBound = j * miniBatchSize
				upperBound = min((j+1)*miniBatchSize, N)

				# step 1 - feedforward - compute error at each layer
				# Input x: Set the corresponding activation 
				# a1 for the input layer.
				# Feedforward: For each l=2,3,…,L
				# compute zl=wlal−1+bl and al=σ(zl).
				#print('\n###########################step 1 - feedforward')
				[z, a] = self.feedForward(x[lowerBound:upperBound, :])

				# step 2 - compute error in output layer neurons
				# Output error δL: Compute the vector δL=∇aC⊙σ′(zL).
				#print('\n###########################step 2 - output layer sigma')
				outputSigma = \
					self.computeErrorInOutputLayerNeurons(a[-1],
						y[lowerBound:upperBound, :], z[-1])

				# step 3 - Backpropagate the error: 
				# For each l=L−1,L−2,…,2 compute δl=((wl+1)Tδl+1)⊙σ′(zl).
				#print('\n###########################step 3 - all layer sigmas')
				allLayersSigma = self.backPropogateError(z, a
					, outputSigma)

				# step 4 - compute derivative of cost wrt weights
				#print('\n###########################step 4 - error gradients')
				errorGradientWRTWeights = self.errorGradientWRTWeights(
					a, allLayersSigma)

				allLayersSigma = \
					self.averageLayerSigmas(allLayersSigma)

				# step 5
				#print('\n###########################step 5 - update weights')
				self.updateNetworkWeightsAndBiases(errorGradientWRTWeights, 
					allLayersSigma, learningRate, l2Lambda,
					len(range(lowerBound, upperBound)))

	def updateNetworkWeightsAndBiases(self, errorGradientWRTWeights,
		errorGradientWRTbiases, learningRate, l2Lambda, m):
		# print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside updateNetworkWeightsAndBiases()')
		
		for i in range(len(errorGradientWRTbiases)):
			errorW = errorGradientWRTWeights[i]
			errorB = errorGradientWRTbiases[i]

			self.weights[i+1] -= learningRate * errorW
			self.biases[i+1] -= learningRate * errorB

	# classify test input
	def classify(self, x):
		#print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside classify()')
		[_, a] = self.feedForward(x)
		return self.oneHotVectorization(a[-1])

	# network evaluation
	def evaluateNetwork(self, x, y_):
		y = self.classify(x)
		print('shape of classification:', y.shape)
		classificationerror = self.classificationError(y, y_)
		print('classification error is:', classificationerror)
		return classificationerror

	# compute classfication error
	def classificationError(self, y, y_):
		#print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside classificationError()')
		yClasses = np.argmax(y, axis=1)
		y_Classes = np.argmax(y_, axis=1)

		diff = yClasses - y_Classes

		return (np.nonzero(diff != 0))[0].shape[0]/y.shape[0]

	# apply sigmoid to output
	def sigmoid(self, input):
		#print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside sigmoid()')
		sigmoidOutput = 1/1 + np.exp(-1 * input)
		# correcting overflow in np.exp
		sigmoidOutput[np.isnan(sigmoidOutput)] = 0
		sigmoidOutput[np.isneginf(sigmoidOutput)] = -1
		sigmoidOutput[np.isposinf(sigmoidOutput)] = 1
		return sigmoidOutput

	# derivative of sigmoid function
	def derivativeOfSigmoid(self, input):
		#print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside derivativeOfSigmoid()')
		sigmoid = self.sigmoid(input)
		return sigmoid * (1 - sigmoid)

	# one hot vectorization
	def oneHotVectorization(self, vector):
		#print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside oneHotVectorization()')
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

	def describeArrayShapesInList(self, list, msgString):
		print('describing list:', msgString)
		for l in list:
			print(msgString, 'shape:', l.shape)

	def averageLayerSigmas(self, layerSigmas):
		for i in range(len(layerSigmas)):
			N, _ = layerSigmas[i].shape
			layerSigmas[i] = np.sum(layerSigmas[i], axis=0)/N
		return layerSigmas

	def saveNetwork(self, directory):
		# save layer sizes
		ironman.writeNumpyArrayToFile(directory, 'layerSizes',
			self.layerSizes)
		# save weights
		ironman.writeNumpyArrayToFile(directory, 'weights',
			self.weights)
		# save biases
		ironman.writeNumpyArrayToFile(directory, 'biases',
			self.biases)

	def loadNetwork(self, directory):
		# load layer sizes
		self.layerSizes = (ironman.readNumpyArrayFromFile(
			directory+'layerSizes.npy')).tolist()
		# set num layers
		self.numLayers = len(layerSizes)
		# load weights
		self.weights = (ironman.readNumpyArrayFromFile(
			directory+'weights.npy'))
		# load biases
		self.biases = (ironman.readNumpyArrayFromFile(
			directory+'biases.npy'))