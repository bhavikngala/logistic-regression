from tensorflow.examples.tutorials.mnist import input_data
from scipy.cluster.vq import kmeans2
import numpy as np

def readMNISTData():
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	return [mnist.train.images, mnist.train.labels,
			mnist.validation.images, mnist.validation.labels,
			mnist.test.images, mnist.test.labels]

# compute cluster centers and labels using kmeans
def applyKmeans2(data, numClusters):
	centroids, labels = kmeans2(data, numClusters, 
		iter=20, minit='points', missing='warn')
	return [centroids, labels]

# compute inverse of spreads for each clusters in the data
def computeClusterSpreadsInvs(data, lbls):
	lbls = np.array(lbls)
	data = np.array(data)

	unique_lbls = np.unique(lbls)
	
	numlbls = unique_lbls.shape[0]
	spreadInvCols = data.shape[1]
	spreadInvs = np.empty([numlbls, spreadInvCols, spreadInvCols])

	for lbl in unique_lbls:
		lbl_indices = np.nonzero(lbls == lbl)
		lbl_cluster = data[lbl_indices]

		var = np.var(lbl_cluster, axis=0)
		spread = var * np.identity(lbl_cluster.shape[1])
		spreadInv = np.linalg.pinv(spread)

		spreadInvs[lbl, :, :] = spreadInv

	return spreadInvs

# compute design matrix of data
def computeDesignMatrixUsingGaussianBasisFunction(data, means, 
	spreadInvs):
	numDataRows = data.shape[0]
	numBasis = means.shape[0]

	designMatrix = np.empty([numDataRows, numBasis])

	rowIndex = 0
	for (mean, spreadInv) in zip(means, spreadInvs):
		distFromMean = data - mean
		firstBasis = np.sum(np.multiply(
			np.matmul(distFromMean, spreadInv)), axis=1)
		firstBasis = np.exp(-0.5 * firstBasis)
		designMatrix[:, rowIndex] = firstBasis
		rowIndex = rowIndex + 1

	return np.insert(designMatrix, 0, 1, axis=1)

# function provided by TA
def computeWeightsUsingSGD(designMatrix, ouputData, learningRate,
	epochs, batchSize, l2Lambda):
	N,M = designMatrix.shape
	weights = np.zeros([1, M])

	for epoch in range(epochs):
		for i in range(int(N/batchSize)):
			lowerBound = i * batchSize
			upperBound = min((i+1)*batchSize, N)

			phi = designMatrix[lowerBound:upperBound, :]

			target = ouputData[lowerBound:upperBound, :]

			ED = np.matmul(sigmoid((np.matmul(phi, weights.T))\
				 - target).T, phi)
			E = (ED + l2Lambda * weights)/batchSize

			weights = weights - learningRate * E

	return weights

def sigmoid(input):
	return 1/(1 + np.exp(-1 * input))

def predictClass(data, weights):
	data = np.insert(data, 0, 1, axis=1)
	return sigmoid((np.sum(np.multiply(data, weights), axis=1)).T)