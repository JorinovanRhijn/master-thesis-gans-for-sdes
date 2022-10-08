# Jorino van Rhijn
# Monte Carlo Simulation of SDEs with GANs

import torch
import pickle
from typing import Any
from data_types import PreProcessing
from nets import Generator, Discriminator


def select_device() -> torch.DeviceObjType:
    return torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')


def pickle_it(data: Any, path: str):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle(path: str):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def count_layers(net: torch.nn.Module):
    '''
    Utility method to return the amount of layers in a NN. Looks for the layers with gradients and counts them\
         if they are not biases.
    '''
    k = 0
    for n, p in net.named_parameters():
        if (p.requires_grad) and ("bias" not in n):
            k += 1
    return k


def standardise(T: torch.Tensor,):
    return (T-T.mean())/T.std()


def input_sample(N: int, C: dict = None, Z: torch.Tensor = None, device: str = 'cpu'):
    '''
    Sample an input batch to the generator, consisting of a random input and condition label if supplied.
    C must be a tensor of size [N,C_dim] containing the condition values with e.g. N the batch size
    '''
    if Z is None:
        Z = torch.randn((N, 1)).to(device)
    else:
        Z = Z.to(device)
    Z = standardise(Z)
    if C is None:
        return Z
    else:
        return torch.cat((Z, C.to(device)), axis=1)


def dict_to_tensor(D: dict, device: torch.DeviceObjType = torch.device('cpu')):
    '''
    Utility function to convert dictionary into a tensor.
    '''
    if D is None:
        pass
    D_ = D.copy()
    if 'indices' in D_.keys():
        D.pop('indices')
    n = len(D_)
    T = torch.tensor(D_.pop(next(iter(D_.keys()))), dtype=torch.float32, device=device).view(-1, 1)
    if n == 1:
        return T
    else:
        for v in D_.values():
            T = torch.cat((T, torch.tensor(v, dtype=torch.float32, device=device).view(-1, 1)), axis=1)
        return T


def make_test_tensor(C_test: dict, N_test: int, device: str = 'cpu'):
    '''
    Given dict of scalars C_test, outputs a tensor of size [N_test,num_conditions]
    '''
    # Get amount of entries in dict
    n = len(C_test)
    T = torch.ones((N_test, n), device=device)
    for i, v in enumerate(C_test.values()):
        assert not hasattr(v, '__len__'), 'Test conditions must be scalars.'
        T[:, i] *= v
    return T


def append_gradients(D: Discriminator, G: Generator, D_grads: list, G_grads: list):
    '''
    Utility function to append gradients to existing lists [[w1, w2, w3,...],[v1,v2,v3,...]], where w and v
    represent different layers.
    '''
    k = 0
    for n, p in D.named_parameters():
        if (p.requires_grad) and ("bias" not in n):
            D_grads[k].append(p.grad.abs().mean().item())
            k += 1
    k = 0
    for n, p in G.named_parameters():
        if (p.requires_grad) and ("bias" not in n):
            G_grads[k].append(p.grad.abs().mean().item())
            k += 1


def preprocess(X_next: torch.Tensor,
               X_prev: torch.Tensor,
               proc_type: PreProcessing = None,
               K: float = 1,
               S_ref: float = None,
               eps: float = 1e-8):
    '''
    Method to compute returns,  logreturns and scaling with strike price K. X_prev must be either scalar or a torch
     tensor of size (N, 1).
    '''
    if proc_type is None:
        return X_next
    assert isinstance(proc_type, PreProcessing),  f'proc_type must be one of \
    {PreProcessing._member_names_}'
    if proc_type is PreProcessing.LOGRETURNS:
        return torch.log(X_next / X_prev + eps)
    elif proc_type is PreProcessing.RETURNS:
        return X_next/X_prev - 1
    elif proc_type is PreProcessing.SCALE_S_REF:
        R = X_next/S_ref-1
        return R


def postprocess(R: torch.Tensor,
                X_prev: torch.Tensor,
                proc_type: PreProcessing = None,
                K: float = 1,
                S_ref: float = None,
                eps: float = 1e-8):
    '''
    Method to invert the pre-processing operation. R should be a torch tensor with returns or the process X itself\
         if proc_type==None.
    '''
    if proc_type is None:
        return R
    assert isinstance(proc_type, PreProcessing),  f'proc_type must be one of \
    {PreProcessing._member_names_}'
    if proc_type is PreProcessing.LOGRETURNS:
        return X_prev*torch.exp(R) + eps
    elif proc_type is PreProcessing.RETURNS:
        return X_prev*(1+R)
    elif proc_type is PreProcessing.LOGRETURNS:
        X_next = torch.abs((R+1)*S_ref)
        return X_next
