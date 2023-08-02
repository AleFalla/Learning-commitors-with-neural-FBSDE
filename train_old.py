import pytorch_lightning as pl
import Utilities as ut
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from model import Hitting_prob_model
import math
from Langevin import Langevin_Dyn
from model import Hitting_prob_model
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from Data_Handler import Data_Handler
from pytorch_lightning.callbacks import StochasticWeightAveraging

# class EMA_Callback(StochasticWeightAveraging):
#     def __init__(self, decay=0.9999):
#         super().__init__()
#         self.decay = decay
    
#     def avg_fn (
#         averaged_model_parameter: torch.Tensor, model_parameter: torch.Tensor, num_averaged: torch.LongTensor
#     ) -> torch.FloatTensor:
#         e = averaged_model_parameter
#         m = model_parameter
#         return self.decay * e + (1. - self.decay) * m
    
torch.set_default_dtype(torch.float32)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class Sine_activation(nn.Module):
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class
    
    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return (torch.sin(input))
    
class B_ID(nn.Module):
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class
    
    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return input + ((input**2+1)**(.5)-1)/2


initial_state = torch.tensor([[-0.6, 1.5],
                              ], device = device)

final_state = torch.tensor([[.6,-0],
                            ], device = device)


mask = torch.tensor([[1., 1.],
                     ]).to(torch.device(device))

mlp_args = {
    'hidden_layers': 5, 
    'width': 256, 
    'input_dim': 3, 
    'activation': Sine_activation(),#nn.SELU(), 
    'norm': False,
    'dropout': False
    }

boundary_args = {
    'reference' : final_state,
    'tolerance' : .2,
    'slope' : .01,
    'mask' : mask,
    'keyword': 'Sigmoid' #alternatives 'Bump', 'Sigmoid', 'Indicator'
}

FBSDE_args = {
    'grad' : False,
    'alpha': 1,
    'beta': 0#1
    }

model_args = {
    'mlp_args': mlp_args, 
    'boundary_args': boundary_args, 
    'FBSDE_args':  FBSDE_args,
    'initial_lr' : 1e-3,
    'loss_type': 'type 2'
}

model = Hitting_prob_model(**model_args)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0)

model.apply(init_weights)

data = Data_Handler(
)

data = data.load_datas_from_files()
training_set = data['train_dataset']
validation_set = data['val_dataset']
print(len(training_set))
train_loader = DataLoader(training_set, batch_size = 50, shuffle = True)
valid_loader = DataLoader(validation_set, batch_size = int(0.25*len(validation_set)), shuffle = True)

checkpoint_callback = ModelCheckpoint(dirpath="checkpoints_final_/", save_top_k=2, monitor="val_loss", save_last = True)

trainer = pl.Trainer(accelerator="gpu", devices=1, callbacks=[checkpoint_callback], max_epochs = 20000, gradient_clip_val = .5, gradient_clip_algorithm = "norm")
trainer.fit(model, train_loader, valid_loader)