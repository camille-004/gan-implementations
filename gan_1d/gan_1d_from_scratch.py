"""
Simple GAN from scratch on a one-dimensional function
From Scratch
"""
# %%
import numpy as np

from core.layers import DenseLayer
from gan_1d.gan_1d import (
    objective, generate_real_samples, generate_fake_samples_generator,
)
from utils.initializers import *
