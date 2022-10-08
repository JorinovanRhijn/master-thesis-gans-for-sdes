import numpy as np
from sde_dataset import SDEDataset
from data_types import Preset


def load_preset(case: Preset, N_train: int = 10_000, N_test: int = 10_000):
    '''
    Method to construct a preset to easily start one of the different cases
    '''
    # Set up the vector of times
    n_steps_vec = np.array([40, 20, 10, 5, 4, 3, 2, 1])
    ts = 2 / n_steps_vec

    # Initialise the dataset, common for each method
    data = SDEDataset()
    data.N = N_train
    data.N_test = N_test
    data.params['s'] = 0
    data.params['t'] = 1

    # Four pre-set cases

    if case is Preset.GBM:
        data.params['mu'] = 0.05
        data.params['sigma'] = 0.2
        data.params['S0'] = 1
    elif case is Preset.CIR_FELLER_SATISFIED:
        data.params['gamma'] = 0.1
        data.params['kappa'] = 0.1
        data.params['S_bar'] = 0.1
        data.params['S0'] = 0.1
        S_vec = np.linspace(0.01, 0.5, 20)
    elif case is Preset.CIR_FELLER_VIOLATED_LOW_GAMMA:
        data.params['gamma'] = 0.3
        data.params['kappa'] = 0.1
        data.params['S_bar'] = 0.1
        data.params['S0'] = 0.1
        S_vec = np.array([1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.02, 0.03, 0.04,
                          0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    elif case is Preset.CIR_FELLER_VIOLATED_HIGH_GAMMA:
        data.params['gamma'] = 0.6
        data.params['kappa'] = 0.1
        data.params['S_bar'] = 0.1
        data.params['S0'] = 0.1
        S_vec = np.concatenate((np.logspace(-6, 0, 12), np.array([0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.5])))
    else:
        raise ValueError('Case {0} not specified in presets. Options are {1}'.format(case, str(Preset._member_names_)))

    # Set up the condition vectors: for the CIR process,
    # a random sample is taken from the Cartesian product of the two conditions
    if case is Preset.GBM:
        # GBM
        data.C = dict(t=ts.repeat(data.N//len(ts))[np.random.permutation(data.N)])
        data.C_test = dict(t=1)
        data.generate_GBM_data()
    else:
        # CIR process
        data.C = dict(S0=S_vec.repeat(data.N//len(S_vec))[np.random.permutation(data.N)],
                      t=ts.repeat(data.N//len(ts))[np.random.permutation(data.N)])
        data.C_test = dict(S0=0.1, t=1)
        data.generate_CIR_data()
    return data
