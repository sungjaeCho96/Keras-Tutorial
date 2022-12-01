import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


def conv_block(
    block_input,
    num_filters=512,
    kernel_size=3,
    dilation_rate=3,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)

    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape

    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = conv_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    out_1 = conv_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = conv_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = conv_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = conv_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = conv_block(x, kernel_size=1)

    return output
