import numpy as np


def constants(shape: tuple, C: int = 1):
    """
    Constant initialization for neural network weights

    :param shape: Shape of neural network weights matrix
    :param C: Constant value
    :return: Initialized weights
    """
    return np.ones(shape) * C


def uniform(shape: tuple, low: float = -0.05, high: float = 0.05):
    """
    Uniform initialization for neural network weights

    :param shape: Shape of neural network weights matrix
    :param low: Uniform distribution lower bound
    :param high: Uniform distribution upper bound
    :return: Initialized weights
    """
    return np.random.uniform(low, high, size=shape)


def gaussian(shape: tuple, mean: float = 0.0, std: float = 1.0):
    """
    Gaussian/normal initialization for neural network weights

    :param shape: Shape of neural network weights matrix
    :param mean: Mean of normal distribution (standard normal by default)
    :param std: Standard deviation of normal distribution
    :return: Initialized weights
    """
    return np.random.normal(mean, std, size=shape)


def lecun_uniform(shape: tuple):
    """
    LeCun uniform initialization for neural network weights (efficient
    backpropagation)

    :param shape: Shape of neural network weights matrix
    :return: Initialized weights
    """
    limit = np.sqrt(3.0 / float(shape[0]))
    return np.random.uniform(-limit, limit, size=shape)


def lecun_normal(shape: tuple):
    """
    LeCun normal initialization for neural network weights (efficient
    backpropagation)

    :param shape: Shape of neural network weights matrix
    :return: Initialized weights
    """
    limit = np.sqrt(1.0 / float(shape[0]))
    return np.random.normal(0.0, limit, size=shape)


def xavier_uniform(shape: tuple):
    """
    Glorot/Xavier uniform initialization for neural network weights

    :param shape: Shape of neural network weights matrix
    :return: Initialized weights
    """
    limit = np.sqrt(6.0 / float(sum(shape)))  # Average # inputs and # outputs
    return np.random.uniform(-limit, limit, size=shape)


def xavier_normal(shape: tuple):
    """
    Glorot/Xavier normal initialization for neural network weights

    :param shape: Shape of neural network weights matrix
    :return: Initialized weights
    """
    limit = np.sqrt(2.0 / float(sum(shape)))
    return np.random.normal(0.0, limit, size=shape)


def he_uniform(shape: tuple):
    """
    He uniform initialization for neural network weights

    :param shape: Shape of neural network weights matrix
    :return: Initialized weights
    """
    limit = np.sqrt(6.0 / float(shape[0]))
    return np.random.uniform(-limit, limit, size=shape)


def he_normal(shape: tuple):
    """
    He normal initialization for neural network weights

    :param shape: Shape of neural network weights matrix
    :return: Initialized weights
    """
    limit = np.sqrt(2.0 / float(shape[0]))
    return np.random.normal(0.0, limit, size=shape)


def initialize_weights(kernel_initialization: str, shape: tuple, **kwargs):
    """
    Initialize weights according to specified initialization technique

    :param kernel_initialization: Initialization method to use
    :param shape: Shape of neural network weights matrix
    :return: Initialized weights
    """
    if kernel_initialization == 'constants':
        return constants(shape, C=kwargs['C'])
    elif kernel_initialization == 'uniform':
        return uniform(shape, low=kwargs['low'], high=kwargs['high'])
    elif kernel_initialization in ('normal', 'gaussian'):
        return gaussian(shape, mean=kwargs['mean'], std=kwargs['std'])
    elif kernel_initialization == 'lecun_uniform':
        return lecun_uniform(shape)
    elif kernel_initialization == 'lecun_normal':
        return lecun_normal(shape)
    elif kernel_initialization == 'xavier_uniform':
        return xavier_uniform(shape)
    elif kernel_initialization == 'xavier_normal':
        return xavier_uniform(shape)
    elif kernel_initialization == 'he_uniform':
        return he_uniform(shape)
    elif kernel_initialization == 'he_normal':
        return he_normal(shape)
    else:
        raise ValueError('This initialization is not supported.')
