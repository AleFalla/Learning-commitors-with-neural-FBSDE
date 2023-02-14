import os
from torch import optim, nn, utils, Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
import Utilities as ut
import torch
from Langevin import Langevin_Dyn
from FBSDE_Helper import FBSDE
from Boundary_Conditions import Boundary_functions

class MLP_sigmoided(torch.nn.Module):

    def __init__(self, hidden_layers = 2, width = 20, input_dim = 10, activation = nn.Sigmoid(), batch_norm = False, dropout = False):
        super(MLP_sigmoided, self).__init__()
        
        layers = []
        if batch_norm:
            layers.append(nn.LayerNorm([input_dim]))
        if dropout:
            layers.append(nn.Dropout(p=0.1))
        layers.append(nn.Linear(input_dim, width))
        layers.append(activation)

        for i in range(0, hidden_layers):
            layers.append(nn.Linear(width,width))
            if batch_norm:
                layers.append(nn.LayerNorm([width]))
            if dropout:
                layers.append(nn.Dropout(p=0.1))
            layers.append(activation)

        layers.append(nn.Linear(width,1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
        

    def forward(self, x):

        return self.model(x)#torch.clamp(1e-10 + 0.5*(1+self.model(x)), min = 0., max = 1.)


class Hitting_prob_model(pl.LightningModule):
    def __init__(self, 
    mlp_args, 
    boundary_args, 
    FBSDE_args,
    normalization_dict = None, 
    ):
        """
        Arguments:
            mlp_args: [Dict] args for the mlp creation, e.g. 
                    {hidden_layers: 2, width: 20, input_dim: 10, activation: nn.Sigmoid(), batch_norm: False}
            potential_n_args: [Dict] args for the potential, e.g. 
                    {potential: function, gamma: 10, masses: masses of the particles, d: dimensionality}
            mean: [Tensor, (1, ph_space_dim + 2)] center of the box defining the problem (ph_space, temperature, time)
            deviation: [Tensor, (1, ph_space_dim + 2)] width of the box defining the problem (ph_space, temperature, time)
            alpha : [float] coefficient for the FP equation loss
            beta: [float] coefficient fot the boundary conditions loss
        """
        super().__init__()
        self.model = MLP_sigmoided(**mlp_args)
        self.FBSDE = FBSDE(**FBSDE_args)
        self.Boundary = Boundary_functions(**boundary_args)
        self.boundary_args = boundary_args
        self.normalization_dict = normalization_dict

        if normalization_dict is not None:
            self.ph_space_means = normalization_dict['ph_space_means'].view(1,-1)
            self.time_mean = normalization_dict['time_mean'].view(1,-1)
            self.ph_space_std = normalization_dict['ph_space_stds'].view(1,-1)
            self.time_std = normalization_dict['time_std'].view(1,-1)
            

    def model_application(self, x, t):
        """
        method for applying the MLP
        """
        x = x.float()
        t = t.float()

        if len(x.size())==3:
            particles = x.size(1)
            batch_size = x.size(0)
            ph_space_size = x.size(2)
            x = torch.cat((x.view(batch_size, particles*ph_space_size), t), dim = 1)
            check = 0

        if len(x.size())==4:
            particles = x.size(2)
            batch_size = x.size(0)
            ph_space_size = x.size(3)
            time_size = x.size(1)
            x = torch.cat((x.view(batch_size, time_size, ph_space_size*particles), t), dim = 2)
            check = 1
            
        if self.normalization_dict is not None:
            
            ph_space_means = self.ph_space_means.repeat(particles, 1).view(1, -1)
            ph_space_std = self.ph_space_std.repeat(particles, 1).view(1, -1)
            means = torch.cat((ph_space_means, self.time_mean), dim = 1).view(-1)
            stds = torch.cat((ph_space_std, self.time_std), dim = 1).view(-1)

        if self.normalization_dict is not None:
            x = (x-means)/stds
        
        if check == 1:
            return self.model(x).view(x.size(0), x.size(1), 1)
        
        if check == 0:
            return self.model(x).view(x.size(0), 1)


    def training_step(self, batch, batch_idx):
        """
        Here the batch should be structured as a TensorDataset returnin trajectories 
        information (path index, time index, ph_space+time+wiener index) and the diffusion constant vector per trajectory
        """
        X, time, increments, B = batch        
        X = X.float()   
        Y_star, Z = self.FBSDE.compute_YZ(X, time, self.model_application)
        Y_0 = Y_star[:,0,:]
        Y = self.FBSDE.Y_run(Y_0, Z, increments, B)

        X_N_extra = self.boundary_args['reference'].view(1, X.size(2), -1).repeat(int(X.size(0)*.1), 1, 1)
        X_N = torch.cat((X[:,-1,:,:], X_N_extra), dim = 0)
        t_N = torch.cat((time[:, -1], time[0:int(X.size(0)*.1), -1]), dim = 0).view(-1,1)
        Y_star_N, Z_N = self.FBSDE.compute_YZ(X_N, t_N, self.model_application)

        loss, Y_path_loss, Y_terminal_loss, dY_terminal_loss = self.FBSDE.FBSDE_Loss(Y, Y_star, Y_star_N, Z_N, X_N, self.Boundary.Function)

        self.log("train_loss", loss, on_step = True, on_epoch = True, prog_bar = True, sync_dist = True)
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("my_lr", cur_lr, prog_bar=True, on_step=True, sync_dist = True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Here the batch should be structured with ph_space, temperature, 
        time, boundary condition set, boundary condition labels, and every 
        tensor should be prepared during the 'dataset' creation
        """
        torch.set_grad_enabled(True)
        X, time, increments, B = batch        
        X = X.float()   
        Y_star, Z = self.FBSDE.compute_YZ(X, time, self.model_application)
        Y_0 = Y_star[:,0,:]
        Y = self.FBSDE.Y_run(Y_0, Z, increments, B)

        X_N_extra = self.boundary_args['reference'].view(1, X.size(2), -1).repeat(int(X.size(0)*.1), 1, 1)
        X_N = torch.cat((X[:,-1,:,:], X_N_extra), dim = 0)
        t_N = torch.cat((time[:, -1], time[0:int(X.size(0)*.1), -1]), dim = 0).view(-1,1)
        Y_star_N, Z_N = self.FBSDE.compute_YZ(X_N, t_N, self.model_application)
        loss, Y_path_loss, Y_terminal_loss, dY_terminal_loss = self.FBSDE.FBSDE_Loss( Y, Y_star, Y_star_N, Z_N, X_N, self.Boundary.Function)

        print('LOSS:', loss.item())
        print('PATH LOSS:', Y_path_loss.item())
        print('TERMINAL LOSS:', Y_terminal_loss.item())
        print('TERMINAL GRAD LOSS:', dY_terminal_loss.item())
        
        self.log("val_loss", loss, on_step = True, on_epoch = True, prog_bar = True, sync_dist = True)
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("my_lr", cur_lr, prog_bar=True, on_step=True, sync_dist = True)
        self.trainer.optimizers[0].zero_grad()
        return loss
    

    def test_step(self, batch, batch_idx):
        """
        Here the batch should be structured with ph_space, temperature, 
        time, boundary condition set, boundary condition labels, and every 
        tensor should be prepared during the 'dataset' creation
        """
        torch.set_grad_enabled(True)
        X, time, increments, B = batch     
        X = X.float()  
        Y_star, Z = self.FBSDE.compute_YZ(X, time, self.model_application)
        Y_0 = Y_star[:,0,:]
        Y = self.FBSDE.Y_run(Y_0, Z, increments, B)

        X_N_extra = self.boundary_args['reference'].view(1, X.size(2), -1).repeat(int(X.size(0)*.1), 1, 1)
        X_N = torch.cat((X[:,-1,:,:], X_N_extra), dim = 0)
        t_N = torch.cat((time[:, -1], time[0:int(X.size(0)*.1), -1]), dim = 0).view(-1,1)
        Y_star_N, Z_N = self.FBSDE.compute_YZ(X_N, t_N, self.model_application)

        loss, Y_path_loss, Y_terminal_loss, dY_terminal_loss = self.FBSDE.FBSDE_Loss( Y, Y_star, Y_star_N, Z_N, X_N, self.Boundary.Function)

        self.log("test_loss", loss, on_step = True, on_epoch = True, prog_bar = True, sync_dist = True)
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("my_lr", cur_lr, prog_bar=True, on_step=True, sync_dist = True)
        self.trainer.optimizers[0].zero_grad()
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = {"scheduler": ReduceLROnPlateau(optimizer, factor = 0.9, patience = 10), "monitor": "val_loss"}
        return [optimizer], scheduler
