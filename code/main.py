import multiclass_logistic_regression
import singleHiddenLayerBackPropTF
import deep_networks_cnn

def main():
	print('\n\n\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Executing MultiClassLogisticRegression')
	multiclass_logistic_regression.main()

	print('\n\n\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Executing SingleHiddenLayerBackPropNeuralNetwork')
	singleHiddenLayerBackPropTF.main()

	print('\n\n\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Executing ConvolutionalNeuralNetwork')
	deep_networks_cnn.main()

if __name__ == "__main__":
	main()