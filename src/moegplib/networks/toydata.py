""" network.py contains scripts that define the neural network.
In particular, we use laplace model
"""
import numpy as np
import torch
import torch.nn.functional as F

from torch.optim import Adam
from torch import nn

from moegplib.utils.logger import SaveAndLoad, BoardLogger

torch.set_default_dtype(torch.double)


class SnelsonPrimeNet(torch.nn.Module):
    """ We define a single layer MLP for snelson dataset.
    The class is made very specific way that we did not mean to provide a generic class (thus fix layer to 1).
    """
    def __init__(self, D_in=21, H=200, D_out=7, n_layers=1, activation='tanh', transfer_off=False):
        """ 
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(SnelsonPrimeNet, self).__init__()
        self.trans_off = transfer_off
        self.activation = activation
        
        self.fc_in = nn.Linear(D_in, H) 
        self.hidden_layers = nn.ModuleList([nn.Linear(H, H) for _ in range(n_layers - 2)])
        self.fc_out = nn.Linear(H, D_out)
        
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        if self.trans_off:
            x = self.fc_in(x)
        else:
            x = self.transfer(self.fc_in(x))
        for layer in self.hidden_layers:
            x = self.transfer(layer(x))
        return self.fc_out(x)
    
    @property
    def transfer(self):
        if self.activation == 'tanh':
            return torch.tanh
        elif self.activation == 'relu':
            return F.relu
        elif self.activation == 'sigmoid':
            return torch.sigmoid
        elif self.activation == 'elu':
            return F.elu   