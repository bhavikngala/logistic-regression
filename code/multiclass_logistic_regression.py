import numpy as np
import helper as dumbledore

def buildMultiClassClassifier(train_data, train_lbl,
	vali_data, vali_lbl, numBasis):
	return None

def buildBinaryClassification(train_data, train_lbl,
	vali_data, vali_lbl, numBasis):
	# step 1 - kmeans - find centres and labels
	[centroids, lbls] = dumbledore.applyKmeans2(train_data,
		numBasis)

	# step 2 - compute cluster spread inverses
	clusterSpreadInvs = dumbledore.computeClusterSpreadsInvs(
		train_data, lbls)

	print(clusterSpreadInvs.shape)

	# step 3 - compute design matrix
	# designMatrix = dumbledore.computeDesignMatrix()

def main():
	[mnist_train_img, mnist_train_lbl, mnist_validation_img, \
	mnist_validation_lbl, mnist_test_img, mnist_test_lbl] = \
	dumbledore.readMNISTData()

	buildBinaryClassification(mnist_train_img, mnist_train_lbl,
		mnist_validation_img, mnist_validation_lbl, 10)

if __name__ == '__main__':
	main()