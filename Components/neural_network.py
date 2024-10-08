import numpy as np
from Components.activations import *

class NeuralNetwork:
    def __init__(self, layers, activations):
        # Initialize the network with the given layers and activation functions
        self.layers = layers  # List of layer sizes
        self.activations = activations  # List of activation functions
        self.weights = []
        self.biases = []

        # Initialize weights and biases for each layer
        for i in range(len(layers) - 1):
            weight_matrix = np.random.randn(layers[i], layers[i + 1]) * 0.1
            bias_vector = np.random.randn(1, layers[i + 1]) * 0.1
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def activate(self, x, activation):
        if activation == 'sigmoid':
            return sigmoid(x)
        elif activation == 'tanh':
            return tanh(x)
        elif activation == 'relu':
            return relu(x)
    
    def activate_derivative(self, x, activation):
        if activation == 'sigmoid':
            return sigmoid_derivative(x)
        elif activation == 'tanh':
            return tanh_derivative(x)
        elif activation == 'relu':
            return relu_derivative(x)

    def feedforward(self, x):
        # Store layer activations and z values
        self.layer_activations = [x]
        self.layer_z = []

        # Forward propagation
        for i in range(len(self.weights)):
            z = np.dot(self.layer_activations[-1], self.weights[i]) + self.biases[i]
            a = self.activate(z, self.activations[i])

            self.layer_z.append(z)
            self.layer_activations.append(a)
        
        return self.layer_activations[-1]

    def backpropagate(self, x, y, learning_rate):
        # Perform a forward pass first
        output = self.feedforward(x)

        # Backpropagation
        deltas = []
        error = y - output  # Output layer error
        delta = error * self.activate_derivative(output, self.activations[-1])  # delta for output layer
        deltas.append(delta)

        # Compute deltas for hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            delta = np.dot(deltas[-1], self.weights[i + 1].T) * self.activate_derivative(self.layer_activations[i + 1], self.activations[i])
            deltas.append(delta)
        
        # Reverse the deltas list
        deltas.reverse()

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] += np.dot(self.layer_activations[i].T, deltas[i]) * learning_rate
            self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * learning_rate

    def train(self, x, y, epochs=10000, learning_rate=0.01):
        for epoch in range(epochs):
            self.backpropagate(x, y, learning_rate)
            if epoch % 1000 == 0:
                loss = np.mean((y - self.feedforward(x)) ** 2)
                print(f"Epoch {epoch}, Loss: {loss}")
