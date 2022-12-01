from utils.config_parser import Configure

config = Configure()
config.set_config("config_001")

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import keras
from utils.Warmup import WarmUpCosine
from dataset.dataset_util import get_dataset_fn
from dataset.augmentation import get_augmentation_layers
from model.model import create_vit_classifier

# Setting seed for reprodcibility
SEED = 42
keras.utils.set_random_seed(SEED)

NUM_CLASSES = 100
INPUT_SHAPE = (32, 32, 3)

dataset_fn = get_dataset_fn("cifar10")  # 나중에는 input 을 입력 받아서 사용할 수 있음

(x_train, y_train), (x_test, y_test) = dataset_fn.load_data()

print(f"x_train shape : {x_train.shape} - y_train shape : {y_train.shape}")
print(f"x_test shape : {x_test.shape} - y_test shape : {y_test.shape}")

augmentation_layers = get_augmentation_layers(config.image_size)
augmentation_layers.layers[0].adapt(x_train)

def run_expreiment(model):
    total_steps = int((len(x_train) / config.batch_size) * config.epochs)
    warmup_epoch_percentage = 0.1
    warmup_steps = int(total_steps * warmup_epoch_percentage)
    scheduled_lrs = WarmUpCosine(
        learning_rate_base=config.lr,
        total_steps=total_steps,
        warmup_learning_rate=0.0,
        warmup_steps=warmup_steps
    )

    optimizer = tfa.optimizers.AdamW(
        learning_rate=config.lr, weight_decay=config.weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="Accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_split=0.1,
    )

    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test, batch_size=config.batch_size)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history


if __name__ == "__main__":
    vit = create_vit_classifier(INPUT_SHAPE, NUM_CLASSES, vanilla=True, augmentation=augmentation_layers)
    history = run_expreiment(vit)
    