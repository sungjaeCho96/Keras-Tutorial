from tensorflow.python.keras import layers

# from tensorflow.keras import layers


class ShiftedPatchTokenization(layers.Layer):
    def __init__(
        self,
        image_size,
        patch_size,
        num_patches,
        projection_dim,
        vanilla=False,
        **kwargs
    ):
        self.test = image_size
