import torch
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader
from model import Hitting_prob_model
from Langevin import Langevin_Dyn
from pytorch_lightning.callbacks import ModelCheckpoint
from Data_Handler import Data_Handler

torch.set_default_dtype(torch.float32)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class Sine_activation(nn.Module):
    def __init__(self):
        '''
        Initializes the Sine_activation module.
        '''
        super().__init__()  # Init the base class

    def forward(self, input):
        '''
        Forward pass of the Sine_activation function.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the Sine activation function.
        '''
        return torch.sin(input)


class B_ID(nn.Module):
    def __init__(self):
        '''
        Initializes the B_ID module.
        '''
        super().__init__()  # Init the base class

    def forward(self, input):
        '''
        Forward pass of the B_ID function.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the B_ID function.
        '''
        return input + ((input**2 + 1)**(.5) - 1) / 2


initial_state = torch.tensor([[-0.6, 1.5],
                              ], device=device)

final_state = torch.tensor([[.6, -0],
                            ], device=device)

mask = torch.tensor([[1., 1.],
                     ]).to(torch.device(device))

mlp_args = {
    'hidden_layers': 5,
    'width': 256,
    'input_dim': 3,
    'activation': Sine_activation(),
    'norm': False,
    'dropout': False
}

boundary_args = {
    'reference': final_state,
    'tolerance': .2,
    'slope': .01,
    'mask': mask,
    'keyword': 'Sigmoid'  # alternatives 'Bump', 'Sigmoid', 'Indicator'
}

FBSDE_args = {
    'grad': False,
    'alpha': 1,
    'beta': 0  # 1
}

model_args = {
    'mlp_args': mlp_args,
    'boundary_args': boundary_args,
    'FBSDE_args': FBSDE_args,
    'initial_lr': 1e-3,
    'loss_type': 'type 2'
}

model = Hitting_prob_model(**model_args)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0)


model.apply(init_weights)

data = Data_Handler()

data = data.load_datas_from_files()
training_set = data['train_dataset']
validation_set = data['val_dataset']
print(len(training_set))
train_loader = DataLoader(training_set, batch_size=50, shuffle=True)
valid_loader = DataLoader(validation_set, batch_size=int(0.25 * len(validation_set)), shuffle=True)

checkpoint_callback = ModelCheckpoint(dirpath="checkpoints_final_/", save_top_k=2, monitor="val_loss", save_last=True)

trainer = pl.Trainer(accelerator="gpu", devices=1, callbacks=[checkpoint_callback], max_epochs=20000, gradient_clip_val=.5, gradient_clip_algorithm="norm")
trainer.fit(model, train_loader, valid_loader)
