import numpy as np
from helpers import ml_helper as dumbledore
from helpers import image_helper as davinci
from helpers import fileHelper as ironman

def buildMultiClassClassifier(train_data, train_lbl,
	test_data, test_lbl, numBasis, learningRate,
	epochs, batchSize, l2Lambda, loadWeights, writeWeights,
	uspsNumeralsImgs, uspsNumeralsLbls,	uspsTestImgs, uspsTestLabels):

	directory = './../results/mutliclass_lr_take01/'
	filename = 'weights.npy'

	if loadWeights:
		print('@@@@@@@@@@@@@@loading weights from file!!!')
		weights = ironman.readNumpyArrayFromFile(directory + filename)
	else:
		designMatrix = np.insert(train_data, 0, 1, axis=1)
		print('shape of designMatrix', designMatrix.shape)

		weights = dumbledore.computeWeightsUsingStochasticGradientDescentTake2(
			designMatrix, train_lbl, learningRate, epochs, batchSize,
			l2Lambda)

	testBasis = np.insert(test_data, 0, 1, axis=1)

	predictedTestClass = dumbledore.predictClass(testBasis,
		weights)

	testPredictionError = dumbledore.classificationError(
		predictedTestClass, test_lbl)
	print('classification error in MNIST test data:',
		testPredictionError)

	uspsNumeralsImgs = np.insert(uspsNumeralsImgs, 0, 1, axis=1)
	uspsNumeralsImgsPredictedLabels = dumbledore.predictClass(
		uspsNumeralsImgs, weights)
	uspsNumeralsPredictionError = dumbledore.classificationError(
		uspsNumeralsImgsPredictedLabels, uspsNumeralsLbls)
	print('classification error on USPS Numerals folder data:',
		uspsNumeralsPredictionError)

	uspsTestImgs = np.insert(uspsTestImgs, 0, 1, axis=1)
	uspsTestImgsPredictedLabels = dumbledore.predictClass(
		uspsTestImgs, weights)
	uspsTestPredictionError = dumbledore.classificationError(
		uspsTestImgsPredictedLabels, uspsTestLabels)
	print('classification error on USPS test folder data:',
		uspsTestPredictionError)

	if writeWeights == True:
		ironman.writeNumpyArrayToFile(directory, filename, weights)

def main():
	learningRate = 0.5
	epochs = 100
	batchSize = 100
	l2Lambda = 0.1
	numBasis = 10
	loadWeights = False
	writeWeights = False

	
	[mnist_train_img, mnist_train_lbl, mnist_validation_img, \
	mnist_validation_lbl, mnist_test_img, mnist_test_lbl] = \
	dumbledore.readMNISTData()

	# read usps images
	directory = './../data/Numerals'
	[uspsNumeralsImgs, uspsNumeralsLbls] = \
		davinci.readUSPSTrainImagesAndLbls(directory)

	directory = './../data/Test'
	[uspsTestImgs, uspsTestLabels] = \
		davinci.readUSPSTestImagesAndLbls(directory)

	buildMultiClassClassifier(mnist_train_img, mnist_train_lbl,
		mnist_test_img, mnist_test_lbl, numBasis,
		learningRate, epochs, batchSize, l2Lambda, loadWeights,
		writeWeights, uspsNumeralsImgs, uspsNumeralsLbls,
		uspsTestImgs, uspsTestLabels)

if __name__ == '__main__':
	main()