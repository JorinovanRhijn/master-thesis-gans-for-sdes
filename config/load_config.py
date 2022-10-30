import yaml
from data_types import Config, MetaParameters, PlotParameters, TestParameters, TrainParameters, NetParameters


def load_config(fname: str = "./config.yaml") -> Config:
    with open(fname) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    config = Config(net_parameters=NetParameters(**config_dict['net_parameters']),
                    train_parameters=TrainParameters(**config_dict['train_parameters']),
                    test_parameters=TestParameters(**config_dict['test_parameters']),
                    meta_parameters=MetaParameters(**config_dict['meta_parameters']),
                    plot_parameters=PlotParameters(**config_dict['plot_parameters']))
    return config
