from Components.neural_network import NeuralNetwork
import numpy as np

# Input data for AND problem
X_test_and = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])

# Expected output labels for AND problem
Y_test_and = np.array([[0],
                       [0],
                       [0],
                       [1]])

# Initialize a neural network
# For AND, we'll use 2 input neurons, 2 hidden neurons, and 1 output neuron
# ReLU activation for hidden layer, Sigmoid activation for output layer
nn_and = NeuralNetwork(layers=[2, 2, 1], activations=['relu', 'sigmoid'])

# Train the neural network
nn_and.train(X_test_and, Y_test_and, epochs=10000, learning_rate=0.1)

# Test predictions after training
print("Predictions for AND problem after training:")
print(nn_and.feedforward(X_test_and))

# Expected output: values close to [0, 0, 0, 1]
