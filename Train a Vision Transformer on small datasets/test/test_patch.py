import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from dataset.dataset_util import get_dataset_fn
from model.model_utils import ShiftedPatchTokenization
from utils.config_parser import parse_config


def test_fn():
    configures = parse_config("config_001")

    dataset_fn = get_dataset_fn("cifar10")

    (x_train, y_train), (x_test, y_test) = dataset_fn.load_data()

    image = x_train[np.random.choice(range(x_train.shape[0]))]
    resized_image = tf.image.resize(
        tf.convert_to_tensor([image]),
        size=(configures["image_size"], configures["image_size"]),
    )

    # Vanilla patch maker : This takes an image and divides into
    # patches as in the original ViT paper
    num_patches = (configures["image_size"] // configures["patch_size"]) ** 2

    (token, patch) = ShiftedPatchTokenization(
        configures["image_size"],
        configures["patch_size"],
        num_patches,
        configures["projection_dim"],
        vanilla=True,
    )(resized_image / 255.0)
    (token, patch) = (token[0], patch[0])

    n = patch.shape[0]
    count = 1
    plt.figure(figsize=(4, 4))
    for row in range(n):
        for col in range(n):
            plt.subplot(n, n, count)
            count += 1
            image = tf.reshape(
                patch[row][col], (configures["patch_size"], configures["patch_size"], 3)
            )
            plt.imshow(image)
            plt.axis("off")

    print(">>> First Chart")
    plt.show()

    (token, patch) = ShiftedPatchTokenization(
        configures["image_size"],
        configures["patch_size"],
        num_patches,
        configures["projection_dim"],
        vanilla=False,
    )(resized_image / 255.0)
    (token, patch) = (token[0], patch[0])
    n = patch.shape[0]
    shifted_images = ["ORIGINAL", "LEFT-UP", "LEFT-DOWN", "RIGHT-UP", "RIGHT-DOWN"]

    for index, name in enumerate(shifted_images):
        print(name)
        count = 1
        plt.figure(figsize=(4, 4))
        for row in range(n):
            for col in range(n):
                plt.subplot(n, n, count)
                count += 1
                image = tf.reshape(
                    patch[row][col],
                    (configures["patch_size"], configures["patch_size"], 5 * 3),
                )
                plt.imshow(image[..., 3 * index : 3 * index + 3])
                plt.axis("off")
        plt.show()
