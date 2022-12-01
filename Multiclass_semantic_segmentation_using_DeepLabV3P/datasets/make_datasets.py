import tensorflow as tf

from datasets.dataset_config import DatasetConfig

d_config = DatasetConfig()


def read_image(image_path: str, mask: bool = False):
    image = tf.io.read_file(image_path)

    if mask:
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(
            images=image, size=[d_config.image_size, d_config.image_size]
        )
    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(
            images=image, size=[d_config.image_size, d_config.image_size]
        )
        image = image / 127.5 - 1  # pixel 값 normalize

    return image


def load_data(image_list: str, mask_list: str):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)

    return image, mask


def data_generator(image_list, mask_list):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(d_config.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
