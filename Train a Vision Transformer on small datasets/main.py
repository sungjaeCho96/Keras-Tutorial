import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from dataset.dataset_util import get_dataset_fn
from dataset.augmentation import get_augmentation_layers

# Setting seed for reprodcibility
SEED = 42
keras.utils.set_random_seed(SEED)

NUM_CLASSES = 100
INPUT_SHAPE = (32, 32, 3)

dataset_fn = get_dataset_fn("cifar10")  # 나중에는 input 을 입력 받아서 사용할 수 있음

(x_train, y_train), (x_test, y_test) = dataset_fn.load_data()

print(f"x_train shape : {x_train.shape} - y_train shape : {y_train.shape}")
print(f"x_test shape : {x_test.shape} - y_test shape : {y_test.shape}")

input_shape = x_train.shape

augmentation_layers = get_augmentation_layers(input_shape[0])
augmentation_layers.layers[0].adapt(x_train)
