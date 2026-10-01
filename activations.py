import numpy as np
import pandas as pd

#SIGMOID

class Activations:
    @staticmethod
    def sigmoid(x):
        x = np.asarray(x)
        return 1 / (1 + np.exp(-x))

    #TANH

    @staticmethod
    def tanh(x):
        x = np.asarray(x)
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    #RELU

    @staticmethod
    def ReLU(x):
        x = np.asarray(x)
        return np.maximum(0, x)

    @staticmethod
    def parametric_ReLU(x, alpha=0.01):
        x = np.asarray(x)
        if x < 0:
            return alpha * x
        else:
            return x

    @staticmethod
    def leaky_ReLU(x):
        return Activations.parametric_ReLU(x, 0.01)

    @staticmethod
    def ELU(x, alpha=1.0):
        x = np.asarray(x)
        if x < 0:
            return alpha * (np.exp(x) - 1)
        else:
            return x

    @staticmethod
    def SELU(x, alpha=1.67326324, lambda_=1.05070098):
        x = np.asarray(x)
        if x < 0:
            return lambda_ * alpha * (np.exp(x) - 1)
        else:
            return lambda_ * x

    #SOFTMAX

    @staticmethod
    def softmax(x):
        x = np.asarray(x)
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
