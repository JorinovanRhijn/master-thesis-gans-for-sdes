import torch
from typing import Union
from utils import standardise
from data_types import PreProcessing
from nets import Generator, Discriminator


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


def preprocess(X_next: torch.Tensor,
               X_prev: torch.Tensor,
               proc_type: PreProcessing = None,
               S_ref: float = None,
               eps: float = 1e-8):
    '''
    Method to compute returns, logreturns and scaling with strike price K. X_prev must be either scalar or a torch
     tensor of size (N, 1).
    '''
    if proc_type is None:
        return X_next
    assert isinstance(proc_type, PreProcessing), f'proc_type must be one of \
    {PreProcessing._member_names_}'
    if proc_type is PreProcessing.LOGRETURNS:
        assert X_prev is not None, "X_prev must be specified to calculate next return value."
        return torch.log(X_next / X_prev + eps)
    elif proc_type is PreProcessing.RETURNS:
        assert X_prev is not None, "X_prev must be specified to calculate next return value."
        return X_next/X_prev - 1
    elif proc_type is PreProcessing.SCALE_S_REF:
        assert S_ref is not None, "S_ref must be specified to calculate next return value."
        R = X_next/S_ref-1
        return R


def postprocess(R: torch.Tensor,
                X_prev: torch.Tensor,
                proc_type: PreProcessing = None,
                S_ref: float = None,
                eps: float = 1e-8):
    '''
    Method to invert the pre-processing operation. R should be a torch tensor with returns or the process X itself\
         if proc_type==None.
    '''
    if proc_type is None:
        return R
    assert isinstance(proc_type, PreProcessing), f'proc_type must be one of \
    {PreProcessing._member_names_}'
    if proc_type is PreProcessing.LOGRETURNS:
        assert X_prev is not None, "X_prev must be specified to calculate next return value."
        return X_prev*torch.exp(R) + eps
    elif proc_type is PreProcessing.RETURNS:
        assert X_prev is not None, "X_prev must be specified to calculate next return value."
        return X_prev*(1+R)
    elif proc_type is PreProcessing.SCALE_S_REF:
        assert S_ref is not None, "S_ref must be specified to calculate next value."
        X_next = torch.abs((R+1)*S_ref)
        return X_next


def inference_sample(input_sample: torch.Tensor,
                     net: Union[Discriminator, Generator],
                     X_prev: torch.Tensor = None,
                     S_ref: float = None,
                     proc_type: PreProcessing = None,
                     eps: float = 1e-8,
                     to_np: bool = False,
                     raw_output: bool = False,
                     device: torch.DeviceObjType = torch.device('cpu')):

    output = net(input_sample).detach().to(device).view(-1)
    if to_np:
        output = output.numpy()

    if raw_output:
        return output
    else:
        return postprocess(output, X_prev, proc_type, S_ref, eps)
