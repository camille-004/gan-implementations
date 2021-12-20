# %%
import os

import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import (
    load_img, img_to_array
)
from tqdm import tqdm

DATA_DIR = '../data/'


def load_images(_path, size=(256, 512)):
    """
    Load, rescale, split map/satellite images

    :param _path: path of directory from which to load images
    :param size: size of images to load
    :return: list of numpy arrays containing pixels of each image
    """
    _src, _target = [], []

    for file in tqdm(os.listdir(_path)):
        pixels = load_img(os.path.join(_path, file), target_size=size)
        pixels = img_to_array(pixels)
        satellite_img, map_img = pixels[:, :256], pixels[:, 256:]
        _src.append(satellite_img)
        _target.append(map_img)

    all_images = [np.asarray(_src), np.asarray(_target)]
    print(f'Loaded: source images {all_images[0].shape}, '
          f'target images: {all_images[1].shape}')
    return all_images


def save_images(_src_array, _target_array, _out_file):
    """
    Save preprocessed images to memory as a compressed numpy array

    :param _src_array: array of source image pixels
    :param _target_array: array of target image pixels
    :param _out_file: path to which compressed images will be saved
    :return:
    """
    print(f'Saved dataset: {_out_file}')
    np.savez_compressed(_out_file, _src_array, _target_array)


def plot_images(_src_array, _target_array, n_samples=3, offset=0):
    """
    Plot n_samples source and target images

    :param _src_array: input source image
    :param _target_array: input target images
    :param n_samples: number of examples plot
    :param offset: which index image in the dataset to start with
    :return:
    """
    # Plot source images
    for i in range(n_samples):
        plt.subplot(2, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(_src_array[i + offset].astype('uint8'))

    # Plot target images
    for i in range(n_samples):
        plt.subplot(2, n_samples, 1 + i + n_samples)
        plt.axis('off')
        plt.imshow(_target_array[i + offset].astype('uint8'))

    plt.show()


# %%
path = os.path.join(DATA_DIR, 'maps/train')
out_file = os.path.join(DATA_DIR, 'maps/maps_256.npz')

[src, target] = load_images(path)
save_images(src, target, out_file)
# %%
# Load dataset
data = np.load(out_file)
src_images, target_images = data['arr_0'], data['arr_1']
print(f'Loaded: source images {src_images.shape}, '
      f'target images {target_images.shape}')
# %%
# Plot images
plot_images(src_images, target_images, offset=10)
