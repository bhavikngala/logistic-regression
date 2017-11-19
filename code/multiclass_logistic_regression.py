import numpy as np
from helpers import ml_helper as dumbledore
from helpers import image_helper as davinci
from helpers import fileHelper as ironman

def buildMultiClassClassifier(train_data, train_lbl,
	vali_data, vali_lbl, numBasis, learningRate,
	epochs, batchSize, l2Lambda, loadWeights):
	# step 1 - kmeans - find centres and labels
	'''
	print('# step 1 - kmeans - find centres and labels')
	[centroids, lbls] = dumbledore.applyKmeans2(train_data,
		numBasis)

	# step 2 - compute cluster spread inverses
	print('# step 2 - compute cluster spread inverses')
	clusterSpreadInvs = dumbledore.computeClusterSpreadsInvs(
		train_data, lbls)
	print('shape of clusterSpreadInvs', clusterSpreadInvs.shape)
	print('\n\n\nclusterspreads \n', clusterSpreadInvs)

	# step 3 - compute design matrix
	print('# step 3 - compute design matrix')
	designMatrix = \
		dumbledore.computeDesignMatrixUsingGaussianBasisFunction(
			train_data, centroids, clusterSpreadInvs)
	'''

	directory = './../results/mutliclass_lr_take01/'
	filename = 'weights.npy'

	if loadWeights:
		print('@@@@@@@@@@@@@@loading weights from file!!!')
		weights = ironman.readNumpyArrayFromFile(directory + filename)
	else:
		designMatrix = np.insert(train_data, 0, 1, axis=1)
		print('shape of designMatrix', designMatrix.shape)

		# step 4 - compute weights - run gradient descent
		print('# step 4 - compute weights - run gradient descent') 
		weights = dumbledore.computeWeightsUsingStochasticGradientDescentTake2(
			designMatrix, train_lbl, learningRate, epochs, batchSize,
			l2Lambda)
		print('shape of weights', weights.shape)

	# step 5 - predict class for validation data
	print('# step 5 - predict class for validation data')
	'''validationBasis = \
		dumbledore.computeDesignMatrixUsingGaussianBasisFunction(
			vali_data, centroids, clusterSpreadInvs)
	'''
	print('shape of weights:', weights.shape)

	validationBasis = np.insert(vali_data, 0, 1, axis=1)

	print('shape of validation basis:', validationBasis.shape)
	predictedValidationClass = dumbledore.predictClass(validationBasis,
		weights)

	# step 6 - one vectorization of predicted classes
	print('# step 6 - one vectorization of predicted classes')
	predictedValidationClass = \
		dumbledore.representPredictionProbsAsOneHotVector(
			predictedValidationClass)

	# step 7 - calculate validation error
	print('# step 7 - calculate validation error')
	validationPredictionError = dumbledore.classificationError(
		predictedValidationClass, vali_lbl)

	print('classification error is:', validationPredictionError)

	if loadWeights == False:
		ironman.writeNumpyArrayToFile(directory, filename, weights)

def main():
	learningRate = 0.5
	epochs = 100
	batchSize = 100
	l2Lambda = 0.1
	numBasis = 10
	loadWeights = True

	
	[mnist_train_img, mnist_train_lbl, mnist_validation_img, \
	mnist_validation_lbl, mnist_test_img, mnist_test_lbl] = \
	dumbledore.readMNISTData()

	# read usps images
	directory = './../data/Numerals'
	[uspsValiImgs, uspsValiLbls] = \
		davinci.readUSPSTrainImagesAndLbls(directory)

	buildMultiClassClassifier(mnist_train_img, mnist_train_lbl,
		uspsValiImgs, uspsValiLbls, numBasis,
		learningRate, epochs, batchSize, l2Lambda, loadWeights)

if __name__ == '__main__':
	main()