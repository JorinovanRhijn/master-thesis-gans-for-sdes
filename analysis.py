import torch
import matplotlib.pyplot as plt
from data_types import Config
from typing import Union
from nets import Generator, Discriminator
from sample import inference_sample, input_sample


class Analysis:
    def __init__(self):
        pass


class Plot:
    LINES = ['dashed', 'solid', 'dotted', 'dashdot']

    def __init__(self,
                 generator: Generator,
                 config: Config,
                 ):
        self.generator = generator
        self.config = config

    def check_generator(self, condition_dict: dict):
        if condition_dict is not None:
            assert len(condition_dict) == self.generator.c_dim,\
             "Condition dict dimension does not match generator input dimension."
        if self.config.meta_parameters.supervised:
            assert self.generator.c_dim == 1 + len(condition_dict) if (condition_dict is not None) else 1,\
                "Condition dict dimension does not match generator input dimension."

    def get_in_sample(self, noise: torch.Tensor = None):
        in_sample = input_sample(self.config.plot_parameters.noise_samples)

    def get_out_sample(self, raw_output: bool = False):
        output = inference_sample()

    def kde(self, condition_dict: dict = None, raw_output: bool = False):
        pass

    def ecdf(self, condition_dict: dict = None, raw_output: bool = False):
        self.check_generator(condition_dict)

    def iter_plot(self):
        pass
