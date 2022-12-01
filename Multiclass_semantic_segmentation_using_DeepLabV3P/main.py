import os
import tensorflow as tf


from tensorflow import keras
from glob import glob
from datetime import datetime

from datasets.make_datasets import data_generator
from datasets.dataset_config import DatasetConfig
from model.model import DeeplabV3Plus
from inference.inference import *

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)

LOG_DIR = "../tf_log/seg/" + datetime.now().strftime("%Y%m%d-%H%M%S")

d_config = DatasetConfig()


def run():
    train_images = sorted(glob(os.path.join(d_config.data_dir, "Images/*")))[
        : d_config.num_train_images
    ]
    train_masks = sorted(glob(os.path.join(d_config.data_dir, "Category_ids/*")))[
        : d_config.num_train_images
    ]

    val_images = sorted(glob(os.path.join(d_config.data_dir, "Images/*")))[
        d_config.num_train_images : d_config.num_val_images + d_config.num_train_images
    ]
    val_masks = sorted(glob(os.path.join(d_config.data_dir, "Category_ids/*")))[
        d_config.num_train_images : d_config.num_val_images + d_config.num_train_images
    ]

    train_dataset = data_generator(train_images, train_masks)
    val_dataset = data_generator(val_images, val_masks)

    print(f">>> Train Dataset : {train_dataset}")
    print(f">>> Val Dataset : {val_dataset}")

    model = DeeplabV3Plus(
        image_size=d_config.image_size, num_classes=d_config.num_classes
    )

    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=0.001, first_decay_steps=1000
    )
    opt = tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn)

    # def get_lr_metric(opt):
    #     def lr(y_true, y_pred):
    #         return opt._decayed_lr(tf.float32)

    #     return lr

    # lr_metric = get_lr_metric(opt)

    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"], jit_compile=True)

    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=LOG_DIR, profile_batch="10, 15"
    )

    def plot_predictions(epoch, logs):
        file_writer = tf.summary.create_file_writer(LOG_DIR)

        image_tensor_list = []
        overlay_list = []
        prediction_colormap_list = []

        for image_file in val_images[:4]:
            image_tensor = read_image(image_file)
            prediction_mask = infer(image_tensor=image_tensor, model=model)
            prediction_colormap = decode_segmentation_masks(prediction_mask, 20)
            overlay = get_overlay(image_tensor, prediction_colormap)

            image_tensor_list.append(image_tensor)
            overlay_list.append(overlay)
            prediction_colormap_list.append(prediction_colormap)

        figure = plot_samples_matplotlib(
            [image_tensor_list, overlay_list, prediction_colormap_list],
            figsize=(18, 14),
        )

        with file_writer.as_default():
            tf.summary.image("Val_images", plot_to_image(figure), step=epoch)

    visualize_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=plot_predictions
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=100,
        callbacks=[tb_callback, visualize_callback],
    )


if __name__ == "__main__":
    run()
