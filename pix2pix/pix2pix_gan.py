# %%
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
