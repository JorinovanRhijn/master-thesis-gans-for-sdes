import yaml
import os
from ..types import MetaParameters, Activation


def load_config() -> MetaParameters:
    with open(f"{os.path.join(os.path.dirname(os.path.abspath(__file__)), __file__)}") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    global config
    config = MetaParameters(**config_dict)
    return config
