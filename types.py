from enum import Enum
from dataclasses import dataclass


@dataclass
class Preset(Enum):
    CIR_FELLER_VIOLATED_HIGH_GAMMA = 'CIR_Feller_violated_high_gamma'


@dataclass
class Activation(Enum):
    LEAKY_RELU = 'leaky_relu'


@dataclass
class PreProcessing(Enum):
    SCALE_S_REF = 'scale_S_ref'


@dataclass
class MetaParameters:
    c_lr: float  # factor by which the learning rate is divided every cut_lr_evert iterations
    cut_lr_every: int
    epochs: int
    eps: float  # small number added to logreturns and generator output to prevent exactly reaching 0
    batch_size: int
    hidden_dim: int
    preset: str  # 'CIR_Feller_violated_high_gamma'
    N_train: int
    N_test: int
    activation: str
    output_activation: str  # output activation of the generator, discriminator activation currently fixed
    # at sigmoid
    negative_slope: float  # Negative slope for leaky_relu activation
    proc_type: str  # pre-processing type
    beta1: float  # Adam beta_1
    beta2: float  # Adam beta_2
    lr_G: float  # base learning rate of the generator
    lr_D: float  # base learning rate of the discriminator
    n_D: int  # number of discriminator iterations per generator iteration, fixed to 1 for supervised GAN
    supervised: bool  # by default, use a vanilla GAN
    seed: int
    save_figs: bool
    save_iter_plot: bool
    report: bool
    results_path: str
    plot_interval: int


@dataclass
class SDEParams:
    t: float = 1.
    S0: float = 1.
    S0_test: float = 1.
    mu: float = 0.05
    sigma: float = 0.2
    kappa: float = 0.5
    S_bar: float = 1
    gamma: float = 0.1
    s: float = 0.
