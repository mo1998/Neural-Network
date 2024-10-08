from Components.neural_network import NeuralNetwork
import numpy as np

X = np.array([
    [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
    [0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
])

Y = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# Create a neural network
# 30 input neurons, 5 neurons in hidden layer, 3 output neurons (multi-class classification)
nn = NeuralNetwork(layers=[30, 5, 3], activations=['relu', 'sigmoid'])

# Train the network on the dataset
nn.train(X, Y, epochs=5000, learning_rate=0.01)

# Predict after training
print("Predictions after training:")
print(nn.feedforward(X))
