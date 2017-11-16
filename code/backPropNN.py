import numpy as np

# Reference:
# http://neuralnetworksanddeeplearning.com/chap2.html

class backPropNN:

	# class instantiation/constructor
	def __init__(self, layerSizes, learningRate, l2Lambda):
		# number of layers in the network
		self.numLayers = len(layerSizes)
		# number of neurons in each layer
		self.layerSizes = layerSizes
		# learning rate for SGD
		self.learningRate = learningRate
		# growth decay regularizer
		self.l2Lambda = l2Lambda

		# initialize weights and biases for each layer
		# weights for 1st layer are always 1
		# biases for 1st layer are always 0
		self.weights = [np.ones([layerSizes[0], 1])]
		self.biases = [np.zeros([layerSizes[0], 1])]
				
		for i in range(1, len(layerSizes)):
			self.weights.append(np.zeros([layerSizes[i],
				layerSizes[i-1]]))
			self.biases.append(np.zeros([layerSize[i], 1]))

	# for each l=2..L compute
	# 1. zl=wlal−1+bl
	# 2. al=σ(zl)
	def feedForward(self):
		return None

	# compute the vector sigma for output layer
	# δL=∇aC⊙σ′(zL)
	def computeOutputError(self):
		return None

	# For each l=L−1,L−2,…,2 compute δl=((wl+1)Tδl+1)⊙σ′(zl) 
	def backPropogateError(self):
		return None

	# gradient of cost function
	# ∂C/∂wljk=al−1kδlj and ∂C/∂blj=δlj
	def errorGradient(self):
		return None

	# train the network
	def train(self):
		return None

	# classify test input
	def classify(self):
		return None

	# network evaluation
	def evaluateNetwork(self):
		return None

	# compute classfication error
	def classificationError(self):
		return None

	# run stochastic gradient descent
	def stochasticGradientDescent(self):
		return None