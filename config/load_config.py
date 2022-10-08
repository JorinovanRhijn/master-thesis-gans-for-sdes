import yaml
from data_types import Config, MetaParameters, TrainParameters, NetParameters


def load_config() -> Config:
    with open("./config.yaml") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    global config
    config = Config(net_parameters=NetParameters(**config_dict['net_parameters']),
                    train_parameters=TrainParameters(**config_dict['train_parameters']),
                    meta_parameters=MetaParameters(**config_dict['meta_parameters']))
    return config
