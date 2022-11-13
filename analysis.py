from dataclasses import dataclass
from enum import Enum, auto
import torch
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.distributions as smd
from data_types import Config
from typing import Union, List, Tuple, Dict
from data import DatasetBase
from nets import Generator, Discriminator
from utils import make_test_tensor, get_plot_bounds, cond_dict_to_cart_prod
from sample import input_sample, postprocess, preprocess
from KDEpy import FFTKDE


@dataclass
class PlotType(Enum):
    ECDF = auto()
    KDE = auto()


PLOT_TYPE_MAP = dict(
    ecdf=PlotType.ECDF,
    kde=PlotType.KDE
)


class Analysis:
    LINES = ['dashed', 'solid', 'dotted', 'dashdot']

    def __init__(self,
                 dataset: DatasetBase,
                 generator: Generator,
                 config: Config,
                 device: torch.DeviceObjType = torch.device('cpu'),
                 noise_samples: int = None,
                 ):
        self.dataset = dataset
        self.generator = generator
        self.config = config
        if noise_samples is None:
            self.noise_samples = config.plot_parameters.noise_samples
        else:
            self.noise_samples = noise_samples
        self.proc_type = config.meta_parameters.proc_type
        self.device = device

    def check_generator(self, generator: Generator, condition_dict: dict):
        if condition_dict is not None:
            assert len(condition_dict) == generator.c_dim,\
             "Condition dict dimension does not match generator input dimension."

    def infer(self, condition_dict: dict, generator: Generator = None, noise: torch.Tensor = None) -> torch.Tensor:
        if noise is not None:
            assert len(noise) == self.noise_samples,\
                                    "Noise input dimension must match plot noise dimension."
        if generator is None:
            generator = self.generator
        if condition_dict is None:
            condition_tensor = None
        else:
            condition_tensor = make_test_tensor(condition_dict, self.noise_samples)
        in_sample = input_sample(self.noise_samples,
                                 condition_tensor,
                                 noise,
                                 self.device,
                                 )
        output = generator(in_sample).detach().view(-1)
        return output

    def _get_S0(self, condition_dict):
        if condition_dict is not None:
            if 'S0' in condition_dict.keys():
                S0 = condition_dict['S0']
            else:
                S0 = self.dataset.DEFAULT_PARAMS['S0']
        return S0

    def _gen_and_proc_output(self, condition_dict: dict, generator: Generator, raw_output: bool) -> torch.Tensor:
        self.check_generator(generator, condition_dict)
        output = self.infer(condition_dict=condition_dict, generator=generator)
        if not raw_output:
            S0 = self._get_S0(condition_dict)
            output = postprocess(output,
                                 X_prev=S0,
                                 proc_type=self.config.meta_parameters.proc_type,
                                 S_ref=self.config.meta_parameters.S_ref,
                                 eps=self.config.meta_parameters.eps)
        return output

    @staticmethod
    def kde_wrapper(vec, kernel='gaussian', bw='silverman'):
        f = FFTKDE(kernel=kernel, bw=bw).fit(vec)
        return f

    def _get_est_method(self, t: PlotType):
        if t is PlotType.ECDF:
            return smd.ECDF
        elif t is PlotType.KDE:
            return self.kde_wrapper
        else:
            raise ValueError

    def plot(self,
             condition_ranges: Dict[str, Union[List[float], float]] = None,
             generator: Generator = None,
             raw_output: bool = False,
             plot_type: str = "ecdf",
             ax=None):

        plot_type_enum = PLOT_TYPE_MAP.get(plot_type, None)
        estimator = self._get_est_method(plot_type_enum)

        if condition_ranges is None:
            iters = 1
        else:
            all_combs = cond_dict_to_cart_prod(condition_ranges)
            iters = len(all_combs)

        if generator is None:
            generator = self.generator

        x_mins, x_maxs = [], []

        for i in range(iters):
            line = self.LINES[i % len(self.LINES)]
            condition_dict = all_combs[i] if condition_ranges is not None else None
            output_np = self._gen_and_proc_output(condition_dict, generator, raw_output).view(-1).numpy()

            f_est = estimator(output_np)
            params = {**self.dataset.params, **self.dataset.test_params, **condition_dict}\
                if condition_dict is not None else {**self.dataset.params, **self.dataset.test_params}

            if raw_output:
                exact_samples = self.dataset.sample_exact(n=self.noise_samples, params=params).view(-1)
                exact_samples = preprocess(exact_samples,
                                           X_prev=self._get_S0(condition_dict),
                                           proc_type=self.config.meta_parameters.proc_type,
                                           S_ref=self.config.meta_parameters.S_ref,
                                           eps=self.config.meta_parameters.eps)
                f_exact = estimator(exact_samples.numpy())
            else:
                if plot_type_enum is PlotType.ECDF:
                    f_exact = self.dataset.cdf(params=params)
                elif plot_type_enum is PlotType.KDE:
                    f_exact = self.dataset.pdf(params=params)
                else:
                    raise ValueError

            x_min, x_max = get_plot_bounds(output_np)
            x_mins.append(x_min), x_maxs.append(x_maxs)
            x = np.linspace(x_min, x_max, self.config.plot_parameters.n_points)

            if ax is None:
                _, ax = plt.subplots(1, 1, dpi=100)
            ax.plot(x, f_exact(x), 'k', linestyle=line)
            lbl = ""
            lbl_ctr = 0
            for k, v in condition_dict.items():
                lbl += f"{k} = {v}"
                lbl_ctr += 1
                if (len(condition_dict.items()) > 1) and (lbl_ctr != len(condition_dict.items())):
                    lbl += ', '
            ax.plot(x, f_est(x), '-', label=lbl)

        x_min, x_max = min(x_mins), max(x_maxs)

        if raw_output:
            ax.set_xlabel("$R_t$")
        else:
            ax.set_xlabel("$S_t$")
        return ax

    def discriminator_heatmap(self,
                              discriminator: Discriminator,
                              condition_dict: dict,
                              res: int = 500,
                              n_plot: int = 100,
                              xlims: Tuple[float] = None,
                              ylims: Tuple[float] = None):
        pass
