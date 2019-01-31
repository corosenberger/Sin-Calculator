# Sin-Calculator
A basic neural network that calculates the value of the sin function

SimpleNet.py:
-uses numpy arrays for calculation but is interfaced with simple python lists
-intializes with 1 or 3 inputs:
  -layers: a list containing the amount of neurons in each layer (unspecied amount of layers but cannot be changed of initialization)
    -ex: [2,3,4,2] - has 2 inputs, 3 neurons in the first hidden layer, 4 neurons in the second hidden layer, 2 ouputs
    -ex: [44, 100, 33, 14, 10, 8] - 44 inputs, 100 in the first hidden layer, 33 in the second, 14 in the third, 10 in the forth, 8 outputs
  -weights and biases: training data to be retrieved with the readSave function
    -randomly sets weights and biases upon initialization if no training data is passes
-computeOutput:
  -takes one list of the activation of each neuron in the input layer (len(inputList) must equal the amount of neurons in the first layer)
  -saves the activations of each neuron in valuesList and the derivatives of those activations in derivs
    -uses the sigmoid activation function
  -returns the activations of the output layer as a list
-backprop:
  -calculates the gradients of all the weights and biases in the net bases on the values found by calling the computeOutput function
  -adds the calculated gradients to the rest of the previously calculated gradients since the applyGradient function has been called
  -should only be called directly after the computeOutput function has been called
-applyGradient:
  -applies the previously calculated gradients to the weights and biases in the net
  -resets the calculated gradients
  -should only be called after computeOutput and backprop have been called
-getTrainingData:
  -returns the weights and biases as pure lists
-writeSave:
  -saves the net structure and the training data to the specified file name in the current directory
  -will create a new file in the current directory if the file name specified does not exist yet
-readSave:
  -returns the training data saved in fileDir from previous training
  -returns None if fileDir does not exist
  
SinCalculator.py:
  -compressInput:
    -compresses the input so it is between 0 and 1 and can be fed to the net
  -decompressOutput:
    -decompresses the output so it's between -1 and 1 (not 0 and 1 like the sigmoid function returns)
    -flips the sign if the original input was negative
  -main:
    -prints the net's guess of inputVale
    -prints the actual sin(inputValue) to check for accuracy

sincalculatorsave.txt:
  -saved training data the net uses to calculate the sin function:
    -around 95% of all guesses are within 5% of the actual value
    -net is around 10% wrong on average, definitely a skewed right distribution 
