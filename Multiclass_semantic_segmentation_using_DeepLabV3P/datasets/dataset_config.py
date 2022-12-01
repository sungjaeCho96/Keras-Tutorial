class DatasetConfig:
    def __init__(self, **kwargs):
        self._image_size = 256
        self._batch_size = 8
        self._num_classes = 20
        self._data_dir = "./datasets/instance-level-human-parsing/instance-level_human_parsing/Training/"
        self._num_train_images = 1000
        self._num_val_images = 50
        self._colormap = "./datasets/instance-level-human-parsing/instance-level_human_parsing/human_colormap.mat"

    @property
    def image_size(self):
        return self._image_size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data_dir(self):
        return self._data_dir

    @data_dir.setter
    def data_dir(self, directory: str):
        assert type(directory) is str, "directory must be str"
        self._data_dir = directory

    @property
    def num_train_images(self):
        return self._num_train_images

    @property
    def num_val_images(self):
        return self._num_val_images

    @property
    def colormap(self):
        return self._colormap
