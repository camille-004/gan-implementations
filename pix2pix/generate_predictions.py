# %%
import os

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import (
    img_to_array, load_img
)

from pix2pix.pix2pix_gan import load_real_examples


# %%
def load_image(path, size=(256, 256)):
    """
    Load an image with a preferred size

    :param path: Path to input image
    :param size: Preferred input size
    :return: image pixels to be loaded into Pix2Pix model
    """
    pixels = load_img(path, target_size=size)
    pixels = img_to_array(pixels)

    # Scale to [-1, 1]
    pixels = (pixels - 127.5) / 127.5

    # Reshape to 1 sample
    pixels = np.expand_dims(pixels, 0)
    return pixels


# %%
DATA_DIR = 'data/'

# Load dataset
[X_1, X_2] = load_real_examples(os.path.join(DATA_DIR, 'maps/maps_256.npz'))
print(f'Loaded dataset: source - {X_1.shape}, target - {X_2.shape}')

# %%
# Load model
model = load_model(os.path.join('pix2pix/models/', 'model_epoch_100.h5'))
src_image = load_image(os.path.join(DATA_DIR, 'maps/example.jpg'))
print(f'Loaded example source image: {src_image.shape}')
# %%
generated_image = model.predict(src_image)
generated_image = (generated_image + 1) / 2.0

plt.imshow(generated_image[0])
plt.axis('off')
plt.show()
