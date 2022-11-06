import torch
import numpy as np
from data_types import PreProcessing
from data import DatasetBase
from nets import Generator
from utils import make_test_tensor
from sample import postprocess


def make_gan_paths(data: DatasetBase,
                   G: Generator,
                   n_steps: int,
                   C: dict = None,
                   Z: torch.Tensor = None,
                   n_paths: int = None,
                   proc_type: PreProcessing = None,
                   S_ref: float = None,
                   device: str = 'cpu'):
    '''
    Method to make GAN paths: iterates the generator n_steps times to create n_paths paths. \n
    paths = make_gan_paths(data, G , n_steps). One can specify dt, n_paths, params and the increments Z. \n
    '''
    if C is None:
        # By default, select the test condition
        C = data.test_params
    else:
        assert list(data.test_params.keys()) == list(C.keys()), 'Condition keys provided in C must equal the test conditions\
             in the dataset.'
    if proc_type is PreProcessing.SCALE_S_REF:
        assert S_ref is not None, 'Must specify S_ref'
    if 'S0' not in list(C.keys()):
        print('Note: S0 was not in the test condition.')
    # Ensure the test parameters overlap those specified in params
    params = {**data.params, **data.test_params, **C}
    if n_paths is None:
        n_paths = data.n
    if Z is None:
        Z = torch.randn((n_paths, n_steps))
    else:
        assert Z.size(1) == n_steps, 'Increments must be of size n_steps'

    S0 = torch.tensor(params['S0'], dtype=torch.float32)

    S_GAN = torch.zeros((n_paths, n_steps+1))
    S_GAN[:, 0] = S0.view(-1)*torch.ones(n_paths).view(-1)
    C_tensor = make_test_tensor(C, n_paths)

    if 'S0' in C.keys():
        # Initialise the condition tensor
        # Find which condition in C matches S0
        S_index = np.where(np.array(list(C.keys())) == 'S0')[0][0]
        # Initialise the condition tensor with S0
        C_tensor[:, S_index] = S_GAN[:, 0]
        for k in range(n_steps):
            # Cast the previous S into the tensor with conditions C
            C_tensor[:, S_index] = S_GAN[:, k]
            input_G = torch.cat((Z[:, k].view(-1, 1), C_tensor), axis=1).to(device)
            S_GAN[:, k+1] = postprocess(G(input_G).detach().view(-1).cpu(), X_prev=S_GAN[:, k].view(-1),
                                        proc_type=proc_type, S_ref=S_ref).view(-1)
    else:
        # Initialise the condition tensor
        for k in range(n_steps):
            input_G = torch.cat((Z[:, k].view(-1, 1), C_tensor), axis=1).to(device)
            S_GAN[:, k+1] = postprocess(G(input_G).detach().view(-1).cpu(), X_prev=S_GAN[:, k].view(-1),
                                        proc_type=proc_type, S_ref=S_ref).view(-1)

    return S_GAN
