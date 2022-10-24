from tensorflow import keras


def get_augmentation_layers(img_sz: int):
    augmentation = keras.Sequential(
        [
            keras.layers.Normalization(),
            keras.layers.Resizing(img_sz, img_sz),
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(factor=0.02),
            keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        ],
        name="data_augmentation",
    )

    return augmentation
