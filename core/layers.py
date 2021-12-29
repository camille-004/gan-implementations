#%%
import numpy as np

from utils.initializers import *


class DenseLayer:
    """
    A class that implements a Dense layer
    """

    def __init__(self, units, kernel_initializer, input_size, **kwargs):
        self._units = units
        self._kernel_initializer = kernel_initializer
        self._input_size = input_size

        # Weights matrix (n x d)
        self.weights = self.initialize_kernels(**kwargs)

        # Bias vector (d x 1)
        self.bias = np.zeros(self._units)

    def initialize_kernels(self, **kwargs):
        """
        Initialize weights with specified initialization technique

        :return: Initialized weights
        """
        return initialize_weights(
            kernel_initialization=self._kernel_initializer,
            shape=(self._input_size, self._units),
            **kwargs
        )

    def forward(self, _input):
        """
        Basic perceptron forward-propagation

        :param _input: Input from previous layer (n x 1)
        :return: np.array
        """
        return _input @ self.weights + self.bias

    def backward(self, _input, gradient_loss, learning_rate):
        """

        :param _input:
        :param gradient_loss: Gradient of cost function w.r.t. weights
        :param learning_rate:
        :return:
        """
        # Derivative of layer output w.r.t parameters (dE / dW, dE / db)
        # dE / db = dE / d_pred (i.e. dE / db = gradient_loss)
        dE_dW = np.dot(_input.T, gradient_loss)

        # Derivative of error w.r.t input (dE / dX), will act as gradient of
        # loss for previous layer
        dE_dX = np.dot(gradient_loss, self.weights.T)

        # Update rule
        self.weights = self.weights - learning_rate * dE_dW
        self.bias = self.bias - learning_rate * gradient_loss

        return dE_dX
