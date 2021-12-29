# %%
import numpy as np

from utils.activations import get_activation_fn
from utils.initializers import initialize_weights


class Layer:
    """
    Base class for all layers
    """

    def __init__(self, units, input_size):
        self._units = units
        self._input_size = input_size

    def forward(self, _input):
        raise NotImplementedError

    def backward(self, _input, gradient_loss, learning_rate):
        raise NotImplementedError


class Activation(Layer):
    """
    A class that implements an Activation layer
    """

    def __init__(self, activation_name):
        self._activation, self._activation_grad = get_activation_fn(
            activation_name
        )
        self.output = None

    def forward(self, _input, **kwargs):
        """
        Compute activation function on input

        :param _input: input data
        :return: output of activation
        """
        self.output = self._activation(_input, **kwargs)
        return self.output

    def backward(self, _input, gradient_loss, learning_rate, **kwargs):
        """
        Compute dE / dX

        :param _input: input data
        :param gradient_loss: dE / dY
        :param learning_rate: Not used, as there are no learnable parameters
        :return: dE / dX
        """
        return self._activation_grad(_input, **kwargs) * gradient_loss


class DenseLayer(Layer):
    """
    A class that implements a Dense layer
    """

    def __init__(self, units, kernel_initializer, input_size, **kwargs):
        super().__init__(units, input_size)
        self._kernel_initializer = kernel_initializer

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
        Basic perceptron backward-propagation

        :param _input: Input data
        :param gradient_loss: Gradient of cost function w.r.t. weights
        :param learning_rate: Learning rate for backpropagation
        :return: Derivative of error w.r.t. input
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
