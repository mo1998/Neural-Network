import numpy as np

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  #Sigmoid function

def sigmoid_derivative(x):
    return x * (1 - x)  #Derivative of sigmoid 

def tanh(x):
    return np.tanh(x)   #Tanh fucntion

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2  # Derivative of tanh

def relu(x):
    return np.maximum(0, x)  #Relu function

def relu_derivative(x):
    return np.where(x > 0, 1, 0)  # Derivative of ReLU
