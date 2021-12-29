import numpy as np


def leaky_relu(x: np.array, alpha: float = 0.01):
    """
    Compute Leaky ReLU activation function

    :param x: Input data
    :param alpha: Slope for negative values
    :return: Activation output
    """
    return np.maximum(alpha * x, x)


def leaky_relu_grad(x: np.array, alpha: float = 0.01):
    """
    Compute gradient of Leaky ReLU activation function

    :param x: Input data
    :param alpha: Slope for negative values
    :return: Gradient output
    """
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx


def get_activation_fn(activation: str):
    """
    Return instances of activation function specified in input string

    :param activation: Activation name
    :return: tuple, instance of activation function and its gradient
    """
    if activation == 'leaky_relu':
        return leaky_relu, leaky_relu_grad
