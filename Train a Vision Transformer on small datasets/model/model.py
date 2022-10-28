import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils.config_parser import Configure
from model.model_utils import ShiftedPatchTokenization, PatchEncoder

config = Configure()


def mlp(x, hiddne_units, dropout_rate):
    for units in hiddne_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)

    return x


class MultiHeadAttentionLSA(layers.MultiHeadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The traninable temperature term. The initial value is
        # the square root of the key dimension.
        self.tau = tf.Variable(tf.math.sqrt(float(self._key_dim)), trainable=True)

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        query = tf.multiply(query, 1.0 / self.tau)
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_socres_dropout = self._dropout_layer(
            attention_scores, training=training
        )
        attention_output = tf.einsum(
            self._combine_equation, attention_socres_dropout, value
        )

        return attention_output, attention_scores


diag_attn_mask = 1 - tf.eye(config.num_patches)
diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)


def create_vit_classifier(input_shape, num_classes, vanilla=False, **kwargs):
    inputs = layers.Input(shape=input_shape)
    aug_fn = kwargs.get("augmentation", None)
    if aug_fn is not None:
        augmented = aug_fn(inputs)
    else:
        augmented = inputs

    # create patches
    (tokens, _) = ShiftedPatchTokenization(vanilla=vanilla)(augmented)
    # Encode patches
    encoded_patches = PatchEncoder()(tokens)

    for _ in range(config.transformer_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # create a multi-head attention layer
        if not vanilla:
            attention_output = MultiHeadAttentionLSA(
                num_heads=config.num_heads, key_dim=config.projection_dim, dropout=0.1
            )(x1, x1, attention_mask=diag_attn_mask)
        else:
            attention_output = layers.MultiHeadAttention(
                num_heads=config.num_heads, key_dim=config.projection_dim, dropout=0.1
            )(x1, x1)
        # Skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer Normalization
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = mlp(x3, hiddne_units=config.transformer_units, dropout_rate=0.1)
        # Skip connection 2
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP
    features = mlp(representation, hiddne_units=config.mlp_head_units, dropout_rate=0.5)
    # Classify outputs
    logits = layers.Dense(num_classes)(features)
    # Create the keras model
    model = keras.Model(inputs=inputs, outputs=logits)
    return model
