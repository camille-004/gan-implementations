"""
Simple GAN from scratch on a one-dimensional function
"""
# %%
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.utils.vis_utils import plot_model


# %%
def objective(x):
    """
    Example one-dimensional function from which to generate random samples

    :param x: Input x
    :return: float, f(x)
    """
    return -(x + np.sin(x)) * np.exp(-x ** 2.0)


inputs = np.arange(-10.0, 10.0, 0.1)
outputs = [objective(x) for x in inputs]
plt.plot(inputs, outputs)
plt.show()


# %%
def generate_real_samples(n=200):
    """
    Generate uniformly random values between -10 and 10, then calculate
    objective function for each value. Return stack of inputs and outputs
    :param n: Number of samples to generate
    :return: np.array, stack of inputs and outputs
    """
    X_1 = np.random.uniform(-10, 10, n)  # generate random inputs in [-10, 10]
    X_2 = np.array([objective(x) for x in X_1])  # generate outputs
    X_1 = X_1.reshape(n, 1)
    X_2 = X_2.reshape(n, 1)
    X = np.hstack((X_1, X_2))
    y = np.ones((n, 1))
    return X, y


#%%
def build_discriminator(n_inputs=2):
    """
    Build a binary discriminator model

    :param n_inputs: Input dimension for model
    :return: Compiled discriminator model
    """
    model = Sequential()
    model.add(Dense(25,
                    activation='relu',
                    kernel_initializer='he_uniform',
                    input_dim=n_inputs))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


D = build_discriminator()
D.summary()
plot_model(D,
           to_file='1d_gan/discriminator_plot.png',
           show_shapes=True,
           show_layer_names=True)
