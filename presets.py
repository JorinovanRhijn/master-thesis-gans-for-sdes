import numpy as np
from data import GBMDataset, CIRDataset
from data_types import Preset


def load_preset(case: Preset, n_train: int = 10_000, n_test: int = 10_000):
    '''
    Method to construct a preset to easily start one of the different cases
    '''
    # Set up the vector of times
    n_steps_vec = np.array([40, 20, 10, 5, 4, 3, 2, 1])
    ts = 2 / n_steps_vec

    # Initialise the dataset, common for each method

    # Four pre-set cases

    if case is Preset.GBM:
        dataset = GBMDataset(params=dict(n=n_train,
                                         n_test=n_test,
                                         dt=1,
                                         mu=0.05,
                                         sigma=0.2,
                                         S0=1,
                                         ),
                             condition_ranges=dict(t=ts),
                             )
    elif case is Preset.CIR_FELLER_SATISFIED:
        dataset = CIRDataset(params=dict(n=n_train,
                                         n_test=n_test,
                                         dt=1,
                                         gamma=0.1,
                                         kappa=0.1,
                                         S_bar=0.1,
                                         S0=0.1,
                                         ),
                             )
        S_vec = np.linspace(0.01, 0.5, 20)
    elif case is Preset.CIR_FELLER_VIOLATED_LOW_GAMMA:
        S_vec = np.array([1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.02, 0.03, 0.04,
                          0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        dataset = CIRDataset(params=dict(n=n_train,
                                         n_test=n_test,
                                         dt=1,
                                         gamma=0.3,
                                         kappa=0.1,
                                         S_bar=0.1,
                                         S0=0.1,
                                         ),
                             condition_ranges=dict(t=ts, S0=S_vec),
                             )
    elif case is Preset.CIR_FELLER_VIOLATED_HIGH_GAMMA:
        S_vec = np.concatenate((np.logspace(-6, 0, 12), np.array([0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.5])))
        dataset = CIRDataset(params=dict(n=n_train,
                                         n_test=n_test,
                                         dt=1,
                                         gamma=0.6,
                                         kappa=0.1,
                                         S_bar=0.1,
                                         S0=0.1,
                                         ),
                             condition_ranges=dict(t=ts, S0=S_vec),
                             )
    else:
        raise ValueError('Case {0} not specified in presets. Options are {1}'.format(case, str(Preset._member_names_)))
    return dataset
