import io
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from scipy.io import loadmat
from datasets.dataset_config import DatasetConfig
from datasets.make_datasets import read_image

d_config = DatasetConfig()

colormap = loadmat(d_config.colormap)["colormap"]
colormap = colormap * 100
colormap = colormap.astype(np.uint8)


def infer(model, image_tensor):
    predcitions = model.predict(np.expand_dims((image_tensor), axis=0))
    predcitions = np.squeeze(predcitions)
    predcitions = np.argmax(predcitions, axis=2)

    return predcitions


def decode_segmentation_masks(mask, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)

    for l in range(0, n_classes):
        idx = mask == l

        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]

    rgb = np.stack([r, g, b], axis=2)

    return rgb


def get_overlay(image, colored_mask):

    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)

    return overlay


def plot_samples_matplotlib(display_list: list, figsize=(5, 3)):

    image_tensor_list, overlay_list, prediction_colormap_list = display_list

    rows = len(image_tensor_list)
    cols = len(display_list)

    figure, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    for row in range(rows):
        axes[row][0].imshow(
            tf.keras.preprocessing.image.array_to_img(image_tensor_list[row])
        )
        axes[row][1].imshow(
            tf.keras.preprocessing.image.array_to_img(overlay_list[row])
        )
        axes[row][2].imshow(prediction_colormap_list[row])

    # for i in range(len(display_list)):
    #     if display_list[i].shape[-1] == 3:
    #         axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    #     else:
    #         axes[i].imshow(display_list[i])

    return figure


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)

    return image


def plot_predictions(images_list, model, **kwargs):
    log_dict = kwargs.get("log_dict")

    if log_dict is not None:
        file_writer = tf.summary.create_file_writer(log_dict["log_dir"])
        image_type = log_dict["type"]

    image_tensor_list = []
    overlay_list = []
    prediction_colormap_list = []

    for image_file in images_list:
        image_tensor = read_image(image_file)
        prediction_mask = infer(image_tensor=image_tensor, model=model)
        prediction_colormap = decode_segmentation_masks(prediction_mask, 20)
        overlay = get_overlay(image_tensor, prediction_colormap)

        image_tensor_list.append(image_tensor)
        overlay_list.append(overlay)
        prediction_colormap_list.append(prediction_colormap)

        # figure = plot_samples_matplotlib(
        #     [image_tensor, overlay, prediction_colormap], figsize=(18, 14)
        # )

    figure = plot_samples_matplotlib(
        [image_tensor_list, overlay_list, prediction_colormap_list], figsize=(18, 14)
    )

    if log_dict:
        with file_writer.as_default():
            tf.summary.image(image_type, plot_to_image(figure), step=0)
