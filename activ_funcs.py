import numpy as np
from math import exp

# activation func + derivative

def relu(x):
    return (x>0)*x

def relu_prime(x):
    return (x>0)

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

def sigmoid(x):
    return 1/(1+ exp(-x))

def sigmoid_prime(x):
    return exp(x)/((exp(x) - 1)**2)
