import os
import yaml

_CONFIG_PATH = os.path.abspath(
    "./Train a Vision Transformer on small datasets/configures"
)
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


if __name__ == "__main__":
    parse_config("config_001")
