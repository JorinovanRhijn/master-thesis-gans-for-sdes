import os
from enum import Enum, auto
from dataclasses import dataclass


@dataclass
class Preset(Enum):
    GBM = auto()
    CIR_FELLER_SATISFIED = auto()
    CIR_FELLER_VIOLATED_LOW_GAMMA = auto()
    CIR_FELLER_VIOLATED_HIGH_GAMMA = auto()


@dataclass
class PreProcessing(Enum):
    SCALE_S_REF = auto()
    RETURNS = auto()
    LOGRETURNS = auto()


@dataclass
class Activation(Enum):
    RELU = auto()
    LEAKY_RELU = auto()
    TANH = auto()
    SIGMOID = auto()
    SINE = auto()


@dataclass
class DistributionType(Enum):
    CDF = auto()
    PDF = auto()
    PPF = auto()


@dataclass
class SchemeType(Enum):
    GBM_EULER = auto()
    GBM_MILSTEIN = auto()
    CIR_TRUNC_EULER = auto()
    CIR_TRUNC_EULER_FULL = auto()
    CIR_REFL_EULER = auto()
    CIR_MILSTEIN_TRUNC = auto()
    CIR_MILSTEIN_HEFTER = auto()


@dataclass
class NetParameters:
    eps: float  # small number added to logreturns and generator output to prevent exactly reaching 0
    hidden_dim: int
    negative_slope: float  # Negative slope for leaky_relu activation
    activation_str: str
    generator_output_activation_str: str

    @property
    def activation(self) -> Activation:
        _acts = dict(relu=Activation.RELU,
                     leaky_relu=Activation.LEAKY_RELU,
                     tanh=Activation.TANH,
                     sigmoid=Activation.SIGMOID,
                     since=Activation.SINE)
        return _acts.get(self.activation_str, None)

    @property
    def generator_output(self) -> callable:
        _acts = dict(relu=Activation.RELU,
                     leaky_relu=Activation.LEAKY_RELU,
                     tanh=Activation.TANH,
                     sigmoid=Activation.SIGMOID,
                     since=Activation.SINE,
                     )
        return _acts.get(self.activation_str, None)


@dataclass
class TrainParameters:
    n_train: int
    batch_size: int
    c_lr: float  # factor by which the learning rate is divided every cut_lr_evert iterations
    cut_lr_every: int  # number of iterations after which to reduce the learning rate
    epochs: int
    beta1: float  # Adam beta_1
    beta2: float  # Adam beta_2
    lr_G: float  # base learning rate of the generator
    lr_D: float  # base learning rate of the discriminator
    n_D: int  # number of discriminator iterations per generator iteration, fixed to 1 for supervised GAN


@dataclass
class TestParameters:
    n_test: int
    test_condition: dict  # specify a dict with the test parameters


@dataclass
class MetaParameters:
    preset_str: str  # preset parameters for fast testing
    proc_type_str: str  # pre-processing type
    S_ref: float  # variable to scale with if proc_type is set to scale_S_ref
    eps: float  # small increment to add to postprocessing step
    supervised: bool  # by default, use a vanilla GAN
    seed: int  # random seed throughout the training process
    save_figs: bool
    save_iter_plot: bool
    save_log_dict: bool
    plot_interval: int
    seed: int
    output_name: str = "output"
    enable_cuda: bool = True
    conditional_gan: bool = False

    @property
    def default_dir(self):
        _dir = os.path.dirname(__file__)
        return os.path.join(_dir, self.output_name)

    @property
    def preset(self):
        _presets = dict(GBM=Preset.GBM,
                        cir_feller_satisfied=Preset.CIR_FELLER_SATISFIED,
                        cir_feller_violated_low_gamma=Preset.CIR_FELLER_VIOLATED_LOW_GAMMA,
                        cir_feller_violated_high_gamma=Preset.CIR_FELLER_VIOLATED_HIGH_GAMMA,)
        return _presets.get(self.preset_str, None)

    @property
    def proc_type(self):
        _proc_types = dict(scale_S_ref=PreProcessing.SCALE_S_REF,
                           returns=PreProcessing.RETURNS,
                           logreturns=PreProcessing.LOGRETURNS)
        return _proc_types.get(self.proc_type_str, None)


@dataclass
class PlotParameters:
    noise_samples: int
    save_fig: bool
    save_dir: str
    n_points: int


@dataclass
class Config:
    train_parameters: TrainParameters
    test_parameters: TestParameters
    net_parameters: NetParameters
    meta_parameters: MetaParameters
    plot_parameters: PlotParameters


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
