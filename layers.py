import numpy as np


class Dense:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))

        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW = self.x.T @ grad
        self.db = np.sum(grad, axis=0, keepdims=True)

        return grad @ self.W.T

    def parameters(self):
        return [self.W, self.b]

    def gradients(self):
        return [self.dW, self.db]