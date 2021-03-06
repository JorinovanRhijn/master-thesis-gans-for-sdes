# Jorino van Rhijn 
# Monte Carlo Simulation of SDEs with GANs 

import torch
import torch.nn as nn
from GANutils import get_activation

# Basic feed-forward conditioanl GAN architecture 

class Generator(nn.Module):
    def __init__(self,c_dim=None,hidden_dim=200,activation='leaky_relu',output_activation=None,eps=1e-20,**kwargs):
        super(Generator,self).__init__()
        
        # Select activation functions 
        activation = get_activation(activation,**kwargs)
        output_activation = get_activation(output_activation)
        # Specify size of conditional input
        if c_dim is None:
            self.c_dim = 0
        else:
            self.c_dim = c_dim
        # Prevent output itself from attaining 0
        self.eps = 1e-20
        # A very small number is added to prevent logreturns reaching zero. Has only a very small effect 
        # if the generator learns the process S_t itself 
        self.hidden_dim = 200
        self.activation = activation
        self.output_activation = output_activation
        self.input = nn.Linear(1+self.c_dim,self.hidden_dim)
        self.fc1 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim,1)
        
    def forward(self,x):
        x = self.activation(self.input(x))
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.output_activation(self.output(x)) + self.eps
        return x
        
class Discriminator(nn.Module):
    def __init__(self,c_dim=None,hidden_dim=200,activation='leaky_relu',**kwargs):
        super(Discriminator,self).__init__()

        # Select activation function
        activation = get_activation(activation,**kwargs)

        # Specify size of conditional input
        if c_dim is None:
            self.c_dim = 0
        else:
            self.c_dim = c_dim
        self.hidden_dim = 200
        self.activation = activation
        self.input = nn.Linear(1+self.c_dim,self.hidden_dim)
        self.fc1 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim,1) 
        
    def forward(self,x):
        x = self.activation(self.input(x))
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = torch.sigmoid(self.output(x))
        return x


def load_Generator(path,c_dim=0,hidden_dim=200,activation='leaky_relu',output_activation=None,device='cpu',**kwargs):
    netG = Generator(c_dim=c_dim,hidden_dim=hidden_dim,activation=activation,output_activation=output_activation,**kwargs).to(device)
    checkpoint_G = torch.load(path)
    netG.load_state_dict(checkpoint_G)
    return netG

def load_Discriminator(path,c_dim=0,hidden_dim=200,activation='leaky_relu',device='cpu',**kwargs):
    netD = Discriminator(c_dim=c_dim,hidden_dim=hidden_dim,activation=activation,**kwargs).to(device)
    checkpoint_D = torch.load(path)
    netD.load_state_dict(checkpoint_D)
    return netD