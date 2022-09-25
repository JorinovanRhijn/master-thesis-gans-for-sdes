# Jorino van Rhijn
# Monte Carlo Simulation of SDEs with GANs

import numpy as np
import torch
import pickle


def get_activation(key, negative_slope=0.1, **kwargs):
    # Check if negative slope is given as argument
    if negative_slope in list(kwargs.keys()):
        negative_slope = kwargs['negative_slope']
    if key is None:
        # return identity function
        return lambda x: x
    choices = ['relu', 'leaky_relu', 'tanh', 'sigmoid', 'sine']
    if key == choices[0]:
        return torch.relu
    elif key == choices[1]:
        return torch.nn.LeakyReLU(negative_slope=negative_slope)
    elif key == choices[2]:
        return torch.tanh
    elif key == choices[3]:
        return torch.sigmoid
    elif key == choices[4]:
        return torch.sin
    elif key is None or key == 'None':
        # Return identity
        return lambda x: x
    else:
        raise ValueError('Activation type not implemented or understood. \n Possible choices are:%s'%str(choices))


def pickle_it(data,path):
    with open(path,'wb') as f:
        pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)


def unpickle(path):
    with open(path,'rb') as f:
        data = pickle.load(f)
    return data


def count_layers(net):
    '''
    Utility method to return the amount of layers in a NN. Looks for the layers with gradients and counts them if they are not biases.
    '''
    k = 0
    for n,p in net.named_parameters():
        if (p.requires_grad) and ("bias" not in n):
            k += 1
    return k


def standardise(T,):
    return (T-T.mean())/T.std()


def input_sample(N,C=None,method=None,Z=None,device='cpu'):
    '''
    Sample an input batch to the generator, consisting of a random input and condition label if supplied. 
    C must be a tensor of size [N,C_dim] containing the condition values with e.g. N the batch size 
    '''
    if Z is None:
        Z = torch.randn((N,1)).to(device)
    else:
        latent = Z.to(device)
    Z = standardise(Z)
    if C is None:
        return Z
    else:
        return torch.cat((Z,C.to(device)),axis=1)


def dict_to_tensor(D,device=torch.device('cpu')):
    '''
    Utility function to convert dictionary into a tensor. 
    '''
    if D is None:
        pass
    D_ = D.copy()
    if 'indices' in D_.keys():
        D.pop('indices')
    n = len(D_)
    T = torch.tensor(D_.pop(next(iter(D_.keys()))),dtype=torch.float32,device=device).view(-1,1)
    if n == 1:
        return T
    else:
        for v in D_.values():
            T = torch.cat((T,torch.tensor(v,dtype=torch.float32,device=device).view(-1,1)),axis=1)
        return T


def make_test_tensor(C_test,N_test,device='cpu'):
    '''
    Given dict of scalars C_test, outputs a tensor of size [N_test,num_conditions]
    '''
    # Get amount of entries in dict 
    n = len(C_test)
    T = torch.ones((N_test,n),device=device)
    for i,v in enumerate(C_test.values()):
        assert not hasattr(v,'__len__'), 'Test conditions must be scalars.'
        T[:,i] *= v
    return T


def append_gradients(D,G,Dgrads,Ggrads):
    '''
    Utility function to append gradients to existing lists [[w1, w2, w3,...],[v1,v2,v3,...]], where w and v 
    represent different layers.
    '''
    k = 0
    for n,p in D.named_parameters():
        if(p.requires_grad) and ("bias" not in n):
            Dgrads[k].append(p.grad.abs().mean().item())
            k += 1
    k = 0
    for n,p in G.named_parameters():
        if(p.requires_grad) and ("bias" not in n):
            Ggrads[k].append(p.grad.abs().mean().item())
            k += 1


def make_GAN_paths(data,G,n_steps,C=None,Z=None,N_paths=None,proc_type=None,device='cpu'):
    '''
    Method to make GAN paths: iterates the generator n_steps times to create N_paths paths. \n
    paths = make_GAN_paths(data,G,n_steps). One can specify dt, N_paths, params and the increments Z. \n
    '''
    assert data.CGAN == True, 'Dataset must have conditional GAN toggle set to True.'
    if C is None:
        # By default, select the test condition 
        C = data.C_test
    else:
        assert list(data.C_test) == list(C), 'Condition keys provided in C must equal the test conditions in the dataset.'
    if 'S0' not in list(C):
        print('Note: S0 was not in the test condition.')
    # Ensure the test parameters overlap those specified in params
    params = {**data.params,**data.C_test,**C}
    if N_paths is None:
        N_paths = data.N
    if Z is None:
        Z = torch.randn((N_paths,n_steps))
    else:
        assert Z.size(1) == n_steps, 'Increments must be of size n_steps'

    S0 = torch.tensor(params['S0'],dtype=torch.float32)

    S_GAN = torch.zeros((N_paths,n_steps+1))
    S_GAN[:,0] = S0.view(-1)*torch.ones(N_paths).view(-1)

    if 'S0' in C.keys():
        # Initialise the condition tensor
        C_tensor = make_test_tensor({**C,**dict(S0=C['S0'][0].item())},N_paths)
        # Find which condition in C matches S0
        S_index = np.where(np.array(list(C))=='S0')[0][0]
        # Initialise the condition tensor with S0
        C_tensor[:,S_index] = S_GAN[:,0]
        for k in range(n_steps):
            # Cast the previous S into the tensor with conditions C
            C_tensor[:,S_index] = S_GAN[:,k]
            input_G = torch.cat((Z[:,k].view(-1,1),C_tensor),axis=1).to(device)
            S_GAN[:,k+1] = postprocess(G(input_G).detach().view(-1).cpu(),X_prev=S_GAN[:,k].view(-1),proc_type=proc_type,S_ref=params['S_bar']).view(-1)
    else:
        # Initialise the condition tensor
        C_tensor = make_test_tensor(C,N_paths)
        for k in range(n_steps):
            input_G = torch.cat((Z[:,k].view(-1,1),C_tensor),axis=1).to(device)
            S_GAN[:,k+1] = postprocess(G(input_G).detach().view(-1).cpu(),X_prev=S_GAN[:,k].view(-1),proc_type=proc_type,S_ref=params['S_bar']).view(-1)

    return S_GAN


def preprocess(X_next,X_prev,proc_type=None,K=1,S_ref=None,eps=1e-8):
    '''
    Method to compute returns, logreturns and scaling with strike price K. X_prev must be either scalar or a torch tensor of size (N,1). 
    '''
    if proc_type is None:
        return X_next
    assert proc_type in ['logreturns','returns','scale_S_ref'], 'proc_type must be one of [logreturns, returns, scale_S_ref].'
    if proc_type == 'logreturns':
        return torch.log(X_next/X_prev+eps)
    elif proc_type == 'returns':
        return X_next/X_prev - 1
    elif proc_type == 'scale_S_ref':
        R = X_next/S_ref-1
        return R


def postprocess(R,X_prev,proc_type=None,K=1,S_ref=None,delta_t=None,eps=1e-8):
    '''
    Method to invert the pre-processing operation. R should be a torch tensor with returns or the process X itself if proc_type==None. 
    '''
    if proc_type is None:
        return R
    assert proc_type in ['logreturns','returns','scale_S_ref'], 'proc_type must be one of [logreturns, returns, scale_S_ref].'
    if proc_type == 'logreturns':
        return X_prev*torch.exp(R)+eps
    elif proc_type == 'returns':
        return X_prev*(1+R)
    elif proc_type == 'scale_S_ref':
        X_next = torch.abs((R+1)*S_ref)
        return X_next
