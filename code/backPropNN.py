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
			self.weights.append(np.zeros([layerSizes[i],
				layerSizes[i-1]]))
			self.biases.append(np.zeros([1, layerSizes[i]]))

	# for each l=2..L compute
	# 1. zl=wlal−1+bl
	# 2. al=σ(zl)
	def feedForward(self, x):
		#print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside feedForward()')
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
		#print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside computeErrorInOutputLayerNeurons()')
		networkOutput = self.oneHotVectorization(networkOutput)
		dC = networkOutput - actualOutputs
		dSigmoid = self.derivativeOfSigmoid(networkZ)
		return dSigmoid

	# For each l=L−1,L−2,…,2 compute δl=((wl+1)Tδl+1)⊙σ′(zl) 
	def backPropogateError(self, z, a, outputSigma):
		#print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside backPropogateError()')
		hiddenLayerSigma = [None] * (self.numLayers - 1)
		hiddenLayerSigma[-1] = outputSigma

		#print('length of sigmas array:', len(hiddenLayerSigma))

		for i in range(len(hiddenLayerSigma) - 2, -1, -1):
			hiddenLayerSigma[i] = \
				(np.matmul(hiddenLayerSigma[i+1], self.weights[i+2])\
					* self.derivativeOfSigmoid(z[i+1]))
			#print('shape of sigma in layer', str(i),
			#	':', hiddenLayerSigma[i].shape)
		return hiddenLayerSigma

	# gradient of cost function
	# ∂C/∂wljk=al−1kδlj and ∂C/∂blj=δlj
	def errorGradientWRTWeights(self, a, sigmas):
		#print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside errorGradientWRTWeights()')
		errorGradientWRTWeights = []

		for aLMinusOne, sigmaL in zip(a[:-1], sigmas):
			'''
			activations for each input will be multiplied by
			neuron errors for that corresponding neuros
			take average for all the w*sigma for inputs in batch
			refer the photo of the writings on green board at home
			'''
			errorGradientWRTWeightsInCurrentLayer = \
				np.zeros([sigmaL.shape[1], aLMinusOne.shape[1]])
			for a,s in zip(aLMinusOne, sigmaL):
				s = np.reshape(s, [s.shape[0], 1])

				errorGradientWRTWeightsInCurrentLayer = \
					errorGradientWRTWeightsInCurrentLayer + (s*a)

			errorGradientWRTWeightsInCurrentLayer = \
				errorGradientWRTWeightsInCurrentLayer/sigmaL.shape[0]

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
		'''print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside updateNetworkWeightsAndBiases()')
		
		print('\n\n\n')
		self.describeArrayShapesInList(errorGradientWRTbiases,
			"layer sigmas")
		print('\n\n\n')
		self.describeArrayShapesInList(errorGradientWRTWeights,
			"error gradients wrt weights")
		print('\n\n\n')
		self.describeArrayShapesInList(self.weights,
			"network weights")
		print('\n\n\n')
		self.describeArrayShapesInList(self.biases,
			"network biases")
		print('\n\n\n')'''

		
		for i in range(len(errorGradientWRTbiases)):
			errorW = errorGradientWRTWeights[i] + \
				((l2Lambda * self.weights[i+1]))/m
			errorB = errorGradientWRTbiases[i] + \
				((l2Lambda * self.biases[i+1]))/m

			self.weights[i+1] = self.weights[i+1] - \
				(learningRate * errorW)
			self.biases[i+1] = self.biases[i+1] - \
				(learningRate * errorB) 

	# classify test input
	def classify(self, x):
		#print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside classify()')
		[_, a] = self.feedforward(x)
		return self.oneHotVectorization(a[-1])

	# network evaluation
	def evaluateNetwork(self, x, y_):
		y = self.classify(x)
		classificationerror = self.classificationError(y, y_)
		print('classification error is:', classificationerror)
		return classificationerror

	# compute classfication error
	def classificationError(self, y, y_):
		#print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside classificationError()')
		error = 0

		yClasses = np.argmax(y, axis=1)
		y_Classes = np.argmax(y_, axis=1)

		diff = yClasses - y_Classes

		return 1 - ((np.nonzero(diff == 0))[0].shape[0]/y.shape[0])

	# apply sigmoid to output
	def sigmoid(self, input):
		#print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside sigmoid()')
		sigmoidOutput = 1/1 + np.exp(-1 * input)
		return sigmoidOutput

	# derivative of sigmoid function
	def derivativeOfSigmoid(self, input):
		#print('@@@@@@@@@@@@@@@@@@@@@@@@@@@inside derivativeOfSigmoid()')
		expInput = np.exp(input)
		return expInput/((1+expInput)*(1+expInput))

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
		ironman.writeNumpyArrayToFile(directory, 'layerSizes.txt',
			self.layerSizes)
		# save weights
		ironman.writeNumpyArrayToFile(directory, 'weights.txt',
			self.weights)
		# save biases
		ironman.writeNumpyArrayToFile(directory, 'biases.txt',
			self.biases)