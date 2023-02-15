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
        return torch.sin(input)

final_state = torch.tensor([[-2.,0.],
                            [-2.,0.],
                            [-2.,0.],
                            [-2.,0.],
                            [-2.,0.],
                            [-2.,0.],
                            [-2.,0.],
                            [-2.,0.]]).cuda()

mask = torch.tensor([[1.,0.],
                     [1.,0.],
                     [1.,0.],
                     [1.,0.],
                     [1.,0.],
                     [1.,0.],
                     [1.,0.],
                     [1.,0.]]).cuda()

mlp_args = {
    'hidden_layers': 5, 
    'width': 256, 
    'input_dim': 17, 
    'activation': nn.LeakyReLU(), 
    'batch_norm': False,
    'dropout': False
    }

boundary_args = {
    'reference' : final_state,
    'tolerance' : .9,
    'slope' : 0.001,
    'mask' : mask,
    'keyword': 'Bump' #alternatives 'Bump', 'Sigmoid'
}

FBSDE_args = {
    'grad' : True,
    'alpha': 1,
    'beta': .01
}

normalization_dict = {
    'ph_space_means': torch.tensor([0,0]).cuda(),
    'time_mean': torch.tensor([0]).cuda(),
    'ph_space_stds': torch.tensor([5,15]).cuda(),
    'time_std': torch.tensor([1]).cuda()
}

model_args = {
    'mlp_args': mlp_args, 
    'boundary_args': boundary_args, 
    'FBSDE_args':  FBSDE_args,
    'normalization_dict': normalization_dict
}

model = Hitting_prob_model(**model_args)

def xavier_init(model):
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        else:
            bound = math.sqrt(6) / math.sqrt(param.shape[0] + param.shape[1])
            param.data.uniform_(-bound, bound)

xavier_init(model)
data = Data_Handler(
)
data = data.load_datas_from_files()
training_set = data['train_dataset']#TensorDataset(data_train, data_B_train)
validation_set = data['val_dataset']#TensorDataset(data_val, data_B_val)
train_loader = DataLoader(training_set, batch_size = 100, shuffle = True)
valid_loader = DataLoader(validation_set, batch_size = len(validation_set))


checkpoint_callback = ModelCheckpoint(dirpath="checkpoints_s/", save_top_k=1, monitor="val_loss")
trainer = pl.Trainer(accelerator="gpu", devices=1, callbacks=[checkpoint_callback], max_epochs = 1000, gradient_clip_val = 2, gradient_clip_algorithm = "norm")
trainer.fit(model, train_loader, valid_loader)
