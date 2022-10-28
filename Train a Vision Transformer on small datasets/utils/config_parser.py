import os
import yaml

_CONFIG_PATH = os.path.abspath("./configures")
_CONFIG_FILES = {
    f"{config[:-5]}": os.path.join(_CONFIG_PATH, config)
    for config in os.listdir(_CONFIG_PATH)
}


def parse_config(file_name=None):
    if file_name is None:
        raise Exception("Please Enter The config file name!!!")

    try:
        with open(_CONFIG_FILES[file_name]) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        return params

    except Exception as ex:
        print(f"{str(ex)}")


class Configure(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(Configure, cls).__new__(cls)
        return cls.instance

    def set_config(self, file_name):
        configures = parse_config(file_name)
        try:
            self._buffer_size = configures["buffer_size"]
            self._batch_size = configures["batch_size"]
            self._image_size = configures["image_size"]
            self._patch_size = configures["patch_size"]
            self._num_patches = (self._image_size // self._patch_size) ** 2
            self._lr = configures["lr"]
            self._weight_decay = configures["weight_decay"]
            self._epochs = configures["epochs"]
            self._layer_norm_eps = configures["layer_norm_eps"]
            self._transformer_layers = configures["transformer_layers"]
            self._projection_dim = configures["projection_dim"]
            self._num_heads = configures["num_heads"]
            self._transformer_units = [self._projection_dim * 2, self._projection_dim]
            self._mlp_head_units = configures["mlp_head_units"]
        except Exception as e:
            print(f"Please Check your configure file. : {str(e)}")

    @property
    def buffer_size(self):
        return self._buffer_size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def image_size(self):
        return self._image_size

    @property
    def patch_size(self):
        return self._patch_size

    @property
    def num_patches(self):
        return self._num_patches

    @property
    def lr(self):
        return self._lr

    @property
    def weight_decay(self):
        return self._weight_decay

    @property
    def epochs(self):
        return self._epochs

    @property
    def layer_norm_eps(self):
        return self._layer_norm_eps

    @property
    def transformer_layers(self):
        return self._transformer_layers

    @property
    def projection_dim(self):
        return self._projection_dim

    @property
    def num_heads(self):
        return self._num_heads

    @property
    def transformer_units(self):
        return self._transformer_units

    @property
    def mlp_head_units(self):
        return self._mlp_head_units


if __name__ == "__main__":
    config = Configure()
    config.set_config("config_001")
