# Jorino van Rhijn
# Monte Carlo Simulation of SDEs with GANs

import torch
import pickle
import numpy as np
from typing import Any, Tuple, List, Dict
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


def make_condition_cart_product(condition_ranges: dict, n: int):
    """
    Create a random sample from the Cartesian product of the vectors in condition_ranges
    """
    condition_entries = dict()
    for k, v in condition_ranges.items():
        condition_entries[k] = v.repeat(n // len(v))[np.random.permutation(n)]
    return condition_entries


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
    Given dict of scalars C_test, outputs a tensor of size [N_test, num_conditions]
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


def get_plot_bounds(vec, margin=0.1) -> Tuple:
    x_min = min(vec)
    x_max = max(vec)
    x_min -= margin * abs(x_min)
    x_max += margin * abs(x_max)
    return (x_min, x_max)


def cond_dict_to_cart_prod(d: dict) -> List[Dict[str, float]]:

    def to_dict_singletons(lst: List[Dict[str, float]], key: str):
        return [{key: el} for el in lst]

    def all_comb_two_dicts(l1: List[Dict[str, float]], l2: List[Dict[str, float]]):
        prod_list = []
        for d1 in l1:
            for d2 in l2:
                prod_list.append({**d1, **d2})
        return prod_list

    prod_list = []
    keys = list(d.keys())
    vecs = list(d.values())

    # Handle the trivial case of single vector
    if len(vecs) == 1:
        return d

    ctr_range = list(range(len(vecs)))
    ctr_range.pop(0)

    prod_list = to_dict_singletons(vecs[0], keys[0])
    for i in ctr_range:
        prod_list = all_comb_two_dicts(prod_list, to_dict_singletons(vecs[i], keys[i]))

    return prod_list
