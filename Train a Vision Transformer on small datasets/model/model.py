import tensorflow as tf

from tensorflow import keras
from tensorflow.python.keras import layers
# from tensorflow.keras import layers

class MultiHeadAttentionLSA(layers.MultiHeadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The traninable temperature term. The initial value is
        # the square root of the key dimension.
        self.tau = tf.Variable(tf.math.sqrt(float(self._key_dim)), trainable=True)

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        query = tf.multiply(query, 1.0 / self.tau)
        