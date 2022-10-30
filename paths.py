import torch
import numpy as np
from data_types import PreProcessing
from sde_dataset import SDEDataset
from nets import Generator
from utils import make_test_tensor
from sample import postprocess


def make_GAN_paths(data: SDEDataset,
                   G: Generator,
                   n_steps: int,
                   C: dict = None,
                   Z: torch.Tensor = None,
                   N_paths: int = None,
                   proc_type: PreProcessing = None,
                   device: str = 'cpu'):
    '''
    Method to make GAN paths: iterates the generator n_steps times to create N_paths paths. \n
    paths = make_GAN_paths(data,G,n_steps). One can specify dt, N_paths, params and the increments Z. \n
    '''
    assert data.CGAN, 'Dataset must have conditional GAN toggle set to True.'
    if C is None:
        # By default, select the test condition
        C = data.C_test
    else:
        assert list(data.C_test) == list(C), 'Condition keys provided in C must equal the test conditions\
             in the dataset.'
    if 'S0' not in list(C):
        print('Note: S0 was not in the test condition.')
    # Ensure the test parameters overlap those specified in params
    params = {**data.params, **data.C_test, **C}
    if N_paths is None:
        N_paths = data.N
    if Z is None:
        Z = torch.randn((N_paths, n_steps))
    else:
        assert Z.size(1) == n_steps, 'Increments must be of size n_steps'

    S0 = torch.tensor(params['S0'], dtype=torch.float32)

    S_GAN = torch.zeros((N_paths, n_steps+1))
    S_GAN[:, 0] = S0.view(-1)*torch.ones(N_paths).view(-1)

    if 'S0' in C.keys():
        # Initialise the condition tensor
        C_tensor = make_test_tensor({**C, **dict(S0=C['S0'][0].item())}, N_paths)
        # Find which condition in C matches S0
        S_index = np.where(np.array(list(C)) == 'S0')[0][0]
        # Initialise the condition tensor with S0
        C_tensor[:, S_index] = S_GAN[:, 0]
        for k in range(n_steps):
            # Cast the previous S into the tensor with conditions C
            C_tensor[:, S_index] = S_GAN[:, k]
            input_G = torch.cat((Z[:, k].view(-1, 1), C_tensor), axis=1).to(device)
            S_GAN[:, k+1] = postprocess(G(input_G).detach().view(-1).cpu(), X_prev=S_GAN[:, k].view(-1),
                                        proc_type=proc_type, S_ref=params['S_bar']).view(-1)
    else:
        # Initialise the condition tensor
        C_tensor = make_test_tensor(C, N_paths)
        for k in range(n_steps):
            input_G = torch.cat((Z[:, k].view(-1, 1), C_tensor), axis=1).to(device)
            S_GAN[:, k+1] = postprocess(G(input_G).detach().view(-1).cpu(), X_prev=S_GAN[:, k].view(-1),
                                        proc_type=proc_type, S_ref=params['S_bar']).view(-1)

    return S_GAN
