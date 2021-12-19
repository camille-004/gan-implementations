# %%
import os

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import (
    Activation, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose,
    Dropout, LeakyReLU
)
from keras.models import (
    Input, Model
)
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam


# %%
def build_discriminator(image_shape):
    """
    Implements a 70 x 70 PatchGAN.
    - Takes as input two images that are concatenated together and predicts a
      patch output
    - Optimize with binary cross-entropy
    - Use weighting so that updates have half (0.5) the usual effect, to slow
      down changes to discriminator relative to generator model

    :param image_shape: shape of each input image
    :return: compiled PatchGAN discriminator model
    """
    # Initialize weights
    init = RandomNormal(stddev=0.02)

    # Source image input (tensor, not a layer)
    input_src_img = Input(shape=image_shape)

    # Target image input
    input_target_img = Input(shape=image_shape)

    # Concatenate list of inputs
    merged = Concatenate()([input_src_img, input_target_img])

    # Conv64
    d = Conv2D(
        64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init
    )(merged)
    d = LeakyReLU(alpha=0.2)(d)

    # Conv128
    d = Conv2D(
        128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init
    )(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # Conv256
    d = Conv2D(
        256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init
    )(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # Conv512
    d = Conv2D(
        512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init
    )(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(
        512, (4, 4), padding='same', kernel_initializer=init
    )(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    # Patch layer
    d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)

    # Define model
    model = Model([input_src_img, input_target_img], patch_out)

    # Compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(
        loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5]
    )

    return model


def build_encoder_block(in_layer, n_filters, batch_norm=True):
    """
    Helper function to create blocks of layer for the generator's encoder

    :param in_layer: input layer
    :param n_filters: size of feature map
    :param batch_norm: for conditional batch normalization
    :return: encoder block
    """
    init = RandomNormal(stddev=0.02)

    # Down-sampling layer
    g = Conv2D(
        n_filters, (4, 4), strides=(2, 2), padding='same',
        kernel_initializer=init
    )(in_layer)

    # Conditional batch normalization
    if batch_norm:
        g = BatchNormalization()(g, training=True)

    g = LeakyReLU(alpha=0.2)(g)

    return g


def build_decoder_block(in_layer, skip_in, n_filters, dropout=True):
    """
    Helper function to create blocks of layer for generator's decoder

    :param in_layer: input layer
    :param skip_in: skip connection (i.e., first layer connected to last layer,
                    second to second-last, and so on)
    :param n_filters: size of feature map
    :param dropout: for conditional drop out
    :return: decoder block
    """
    init = RandomNormal(stddev=0.02)

    # Up-sampling layer, "deconvolution" layer
    # Shape of output of convolution --> shape of input, while maintaining
    # connectivity pattern
    g = Conv2DTranspose(
        n_filters, (4, 4), strides=(2, 2), padding='same',
        kernel_initializer=init
    )(in_layer)

    g = BatchNormalization()(g, training=True)

    # Conditional dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)

    # Merge with skip connection
    g = Concatenate()([g, skip_in])

    g = Activation('relu')(g)

    return g


def build_generator(image_shape=(256, 256, 3)):
    """
    Implements a U-Net encoder-decoder generator model.

    :param image_shape: input image shape
    :return: encoder-decoder model
    """
    init = RandomNormal(stddev=0.02)
    input_image = Input(shape=image_shape)

    # Encoder model
    e_1 = build_encoder_block(input_image, 64, batch_norm=False)
    e_2 = build_encoder_block(e_1, 128)
    e_3 = build_encoder_block(e_2, 256)
    e_4 = build_encoder_block(e_3, 512)
    e_5 = build_encoder_block(e_4, 512)
    e_6 = build_encoder_block(e_5, 512)
    e_7 = build_encoder_block(e_6, 512)

    # Bottleneck layer
    b = Conv2D(
        512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init
    )(e_7)
    b = Activation('relu')(b)

    # Decoder model
    d_1 = build_decoder_block(b, e_7, 512)
    d_2 = build_decoder_block(d_1, e_6, 512)
    d_3 = build_decoder_block(d_2, e_5, 512)
    d_4 = build_decoder_block(d_3, e_4, 512, dropout=False)
    d_5 = build_decoder_block(d_4, e_3, 256, dropout=False)
    d_6 = build_decoder_block(d_5, e_2, 128, dropout=False)
    d_7 = build_decoder_block(d_6, e_1, 64, dropout=False)

    # Output
    g = Conv2DTranspose(
        3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init
    )(d_7)
    output_image = Activation('tanh')(g)  # Pixel values in [-1, 1]
    model = Model(input_image, output_image)

    return model


def build_GAN(generator, discriminator, image_shape):
    """
    Connects generator G and discriminator D into composite model.
    - Source image provided as input to D and G
    - Output of G is connected to D as "target" image
    - D predicts likelihood G's output being "real"

    :param generator: pre-defined generator model
    :param discriminator: pre-defined discriminator model
    :param image_shape: input image shape
    :return: compiled GAN model
    """
    # Discriminator is updated as standalone, so weights reused but not
    # trainable
    for layer in discriminator.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False

    source_img = Input(shape=image_shape)
    generator_out = generator(source_img)
    discriminator_out = discriminator([source_img, generator_out])
    model = Model(source_img, [discriminator_out, generator_out])
    opt = Adam(lr=0.0002, beta_1=0.5)

    # Two targets: one indicating that generation is real, and real translation
    # of image
    model.compile(
        loss=['binary_crossentropy', 'mae'], optimizer=opt,
        loss_weights=[1, 100]
    )

    return model


def load_real_examples(f_name):
    """
    Load and prepare training images

    :param f_name: filename of training dataset
    :return: standardized arrays of pixels of source and target images
    """
    data = np.load(f_name)
    X_1, X_2 = data['arr_0'], data['arr_1']

    # Scale from [0, 255] to [-1, 1]
    X_1 = (X_1 - 127.5) / 127.5
    X_2 = (X_2 - 127.5) / 127.5

    return [X_1, X_2]


def generate_real_examples(data, n_samples, patch_shape):
    """
    Select batch of random samples

    :param data: training dataset to unpack
    :param n_samples: input number of samples
    :param patch_shape: input patch shape
    :return: images and target
    """
    train_src, train_target = data

    random_idx = np.random.randint(0, train_src.shape[0], n_samples)
    X_1, X_2 = train_src[random_idx], train_target[random_idx]

    # Generate class labels
    y = np.ones((n_samples, patch_shape, patch_shape, 1))

    return [X_1, X_2], y


def generate_fake_examples(generator, samples, patch_shape):
    """
    Generates fake instances of data

    :param generator: generator model
    :param samples: samples from which to predict
    :param patch_shape: input patch shape
    :return: images and target
    """
    X = generator.predict(samples)
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


def summarize_performance(step, generator, data, n_samples=3):
    [X_real_src, X_real_target], _ = generate_real_examples(data, n_samples, 1)
    X_fake_target, _ = generate_fake_examples(generator, X_real_src, 1)

    # Scale pixels from [-1, 1] to [0, 1]
    X_real_src = (X_real_src + 1) / 2.0
    X_real_target = (X_real_target + 1) / 2.0
    X_fake_target = (X_fake_target + 1) / 2.0

    # Plot real source images
    for i in range(n_samples):
        plt.subplot(n_samples, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(X_real_src[i])

    # Plot generated target image
    for i in range(n_samples):
        plt.subplot(n_samples, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(X_fake_target[i])

    # Plot real target image
    for i in range(n_samples):
        plt.subplot(n_samples, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(X_real_target[i])

    plot_f_name = f'epoch_{step + 1}.png'
    plt.savefig(os.path.join('results/', plot_f_name))
    plt.close()

    model_f_name = f'model_epoch_{step + 1}.h5'
    generator.save(os.path.join('models/', model_f_name))

    print(f'> Saved: {plot_f_name}, {model_f_name}')


def train(discriminator, generator, gan, data, n_epochs=100, n_batches=1):
    """
    Trains pix2pix model.
    - Select a batch of real samples
    - Use generator to generate batch of corresponding fake samples w/ real
      sources
    - Update discriminator with batch of real images and fake images
    - Then, update generator with real sources + class labels, real targets
      (for computing loss)
    - Two loss scores + weighted sum score, weighted sum used to update weights
    - Print loss on each update to console, every 10 epochs

    :param discriminator: defined discriminator
    :param generator: defined generator
    :param gan: composite model (D + G)
    :param data: training data
    :param n_epochs: number of epochs
    :param n_batches: batch size
    :return:
    """
    n_patches = discriminator.output_shape[1]
    src_train, target_train = data
    batch_per_epoch = int(len(src_train) / n_batches)
    n_steps = batch_per_epoch * n_epochs

    for i in range(n_steps):
        [X_real_src, X_real_target], y_real = generate_real_examples(
            data, n_batches, n_patches
        )
        X_fake_target, y_fake = generate_fake_examples(
            generator, X_real_src, n_patches
        )
        discriminator_loss_1 = discriminator.train_on_batch(
            [X_real_src, X_real_target], y_real
        )
        discriminator_loss_2 = discriminator.train_on_batch(
            [X_real_src, X_real_target], y_fake
        )
        generator_loss, _, _ = gan.train_on_batch(
            X_real_src, [y_real, X_real_target]
        )
        print(f'> {i + 1}: {discriminator_loss_1}, '
              f'{discriminator_loss_2}, {generator_loss}')
        if (i + 1) % (batch_per_epoch * 10) == 0:
            summarize_performance(i, generator, data)


#%%
dataset = load_real_examples(os.path.join('data/', 'maps/maps_256.npz'))
print(f'Loaded: sources ({dataset[0].shape}, targets ({dataset[1].shape})')

img_shape = dataset[0].shape[1:]

discriminator_model = build_discriminator(img_shape)
generator_model = build_generator(img_shape)
GAN_model = build_GAN(generator_model, discriminator_model, img_shape)

train(
    discriminator_model, generator_model, GAN_model, dataset, n_epochs=200
)
