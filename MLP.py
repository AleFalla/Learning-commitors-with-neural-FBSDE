import os
import torch
from torch import optim, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, ChainedScheduler, CosineAnnealingLR, CyclicLR
import pytorch_lightning as pl
import Utilities as ut
from Langevin import Langevin_Dyn
from FBSDE_Helper import FBSDE

eps = 1e-7

class Cos_out(nn.Module):
    def __init__(self):
        '''
        Initialize the Cos_out class.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return 0.5*(torch.cos(input**2)+1)

class MLP_sigmoided(torch.nn.Module):
    def __init__(self, 
                 hidden_layers=2, 
                 width=20, 
                 input_dim=10, 
                 activation=nn.Sigmoid(), 
                 input_norm=False,
                 norm=False, 
                 dropout=False):
        '''
        Initialize the MLP_sigmoided class.

        Parameters:
            hidden_layers (int): Number of hidden layers in the MLP.
            width (int): Width of each hidden layer.
            input_dim (int): Dimensionality of the input data.
            activation (torch.nn.Module): Activation function to use between layers.
            input_norm (bool): Whether to apply Layer Normalization to the input data.
            norm (bool): Whether to apply Layer Normalization to hidden layers.
            dropout (bool): Whether to use Dropout between hidden layers.
        '''
        super(MLP_sigmoided, self).__init__()
        
        layers = []
        if input_norm:
            layers.append(nn.LayerNorm(input_dim))
        layers.append(nn.Linear(input_dim, width))
        
        layers.append(activation)
        if dropout:
            layers.append(nn.Dropout(p=0.1))

        for i in range(1, hidden_layers):
            layers.append(nn.Linear(width, width))
            if norm:
                layers.append(nn.LayerNorm(width))
            
            layers.append(activation)
            if dropout:
                layers.append(nn.Dropout(p=0.1))
        
        layers.append(nn.Linear(width, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        '''
        Forward pass of the MLP model.

        Parameters:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: The output of the MLP after passing through all layers.
        '''
        return self.model(x)
