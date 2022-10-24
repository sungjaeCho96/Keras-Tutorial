import tensorflow as tf

_DATASETS = {
    "cifar10": tf.keras.datasets.cifar10,
    "cifar100": tf.keras.datasets.cifar100,
    "mnist": tf.keras.datasets.mnist,
}


def get_dataset_fn(name="mnist"):
    return _DATASETS[name]
