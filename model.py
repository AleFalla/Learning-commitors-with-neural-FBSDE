import os
from torch import optim, nn, utils, Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau, ChainedScheduler, CosineAnnealingLR, CyclicLR
import pytorch_lightning as pl
import Utilities as ut
import torch
from Langevin import Langevin_Dyn
from FBSDE_Helper import FBSDE
from Boundary_Conditions import Boundary_functions
from MLP import *
from multitask_loss import *

def softmax(ten):
    z = ten.abs()
    vec = torch.exp(-z/100)
    vec = vec/vec.sum()
    return vec

def entropy(ten):
    p = ten.abs()/(ten.abs().sum())
    return (-p*torch.log(p)).sum()

class Hitting_prob_model(pl.LightningModule):
    def __init__(self, 
                 mlp_args, 
                 boundary_args, 
                 FBSDE_args,
                 initial_lr=1e-3,
                 loss_type='type 1'
                 ):
        '''
        Initialize the Hitting_prob_model class.

        Parameters:
            mlp_args (dict): Arguments for the MLP creation.
            boundary_args (dict): Arguments for the Boundary_functions class.
            FBSDE_args (dict): Arguments for the FBSDE class.
            initial_lr (float): Initial learning rate for the optimizer.
            loss_type (str): Type of loss to use ('type 1' or 'type 2').
        '''
        super().__init__()
        self.model = MLP_sigmoided(**mlp_args)
        self.fbsde_args = FBSDE_args
        self.FBSDE = FBSDE(**FBSDE_args)
        self.Boundary = Boundary_functions(**boundary_args)
        self.boundary_args = boundary_args
        self.initial_lr = initial_lr
        self.loss_type = loss_type
        self.epochs = 0

    def model_step(self, x, t):
        '''
        Perform a forward pass through the MLP model.

        Parameters:
            x (torch.Tensor): Input data.
            t (torch.Tensor): Time data.

        Returns:
            torch.Tensor: Output of the MLP model.
        '''
        x = x.float()
        t = t.float()
        batch_size = x.size(0)
        particles = x.size(1)
        ph_space_size = x.size(2)
        time = t.view(batch_size, 1, 1).repeat(1, particles, 1)
        x = torch.cat((x, time), dim=-1)
        return self.model(x).prod(-2)

    def model_trajs(self, x, t):
        '''
        Perform a forward pass through the MLP model for trajectory data.

        Parameters:
            x (torch.Tensor): Input data.
            t (torch.Tensor): Time data.

        Returns:
            torch.Tensor: Output of the MLP model for trajectories.
        '''
        x = x.float()
        t = t.float()
        particles = x.size(2)
        batch_size = x.size(0)
        ph_space_size = x.size(3)
        time_size = x.size(1)
        time = t.view(batch_size, time_size, 1, 1).repeat(1, 1, particles, 1)
        x = torch.cat((x, time), dim=-1)
        return self.model(x).prod(-2)

    def type_1_loss(self, batch, verbose=False):
        '''
        Calculate the loss for 'type 1' FBSDE. See https://arxiv.org/abs/1804.07010.

        Parameters:
            batch (tuple): Tuple containing data needed for computing the loss.
            verbose (bool): Whether to print intermediate loss values.

        Returns:
            torch.Tensor: The computed loss.
        '''
        X, time, increments, A, B = batch        
        X = X.float()

        Y_star, Z = self.FBSDE.compute_YZ(X, time, self.model_trajs)
        Y = self.FBSDE.compute_stepwise_Ys(Y_star, Z, increments, B)
        Y_path_loss = self.FBSDE.OLD_Y_loss(Y_star, Y)
        terminal_Y, terminal_Z = self.FBSDE.compute_YZ(X, time[:,-1,0].view(time.size(0), 1, 1).repeat(1,time.size(1), 1), self.model_trajs)
        Y_terminal_loss = self.FBSDE.Y_terminal_loss(terminal_Y.view(-1, Y.size(2)), X.view(-1, X.size(2), X.size(3)), self.Boundary.Function)
        dY_terminal_loss = self.FBSDE.dY_terminal_loss(terminal_Z.view(-1, terminal_Z.size(2), terminal_Z.size(3)), X.view(-1, X.size(2), X.size(3)), self.Boundary.Function)

        loss = Y_path_loss.mean() + self.fbsde_args['alpha']*Y_terminal_loss.mean() + self.fbsde_args['beta']*dY_terminal_loss.mean()

        if verbose:
            return Y_path_loss, Y_terminal_loss, dY_terminal_loss, loss
        else:
            return loss

    def type_2_loss(self, batch, verbose=False):
        '''
        Calculate the loss for 'type 2' FBSDE. See https://arxiv.org/abs/2012.07924 to be honest this is not working very well and is less efficient, my implementation problems of course.

        Parameters:
            batch (tuple): Tuple containing data needed for computing the loss.
            verbose (bool): Whether to print intermediate loss values.

        Returns:
            torch.Tensor: The computed loss.
        '''
        X, time, increments, A, B = batch        
        X = X.float()

        Y_star, Z = self.FBSDE.compute_YZ(X, time, self.model_trajs)
        Y_0 = Y_star[:,0,:]
        Y = self.FBSDE.Y_run(Y_0, Z, increments, B)
        Y_path_loss = self.FBSDE.Y_path_loss(Y, Y_star)
        terminal_Y, terminal_Z = self.FBSDE.compute_YZ(X, time[:,-1,0].view(time.size(0), 1, 1).repeat(1,time.size(1), 1), self.model_trajs)
        Y_terminal_loss = self.FBSDE.Y_terminal_loss(terminal_Y.view(-1, Y.size(2)), X.view(-1, X.size(2), X.size(3)), self.Boundary.Function)
        dY_terminal_loss = self.FBSDE.dY_terminal_loss(terminal_Z.view(-1, terminal_Z.size(2), terminal_Z.size(3)), X.view(-1, X.size(2), X.size(3)), self.Boundary.Function)

        loss = Y_path_loss.mean() + self.fbsde_args['alpha']*Y_terminal_loss.mean() + self.fbsde_args['beta']*dY_terminal_loss.mean()

        if verbose:
            return Y_path_loss, Y_terminal_loss, dY_terminal_loss, loss
        else:
            return loss

    def training_step(self, batch, batch_idx):
        '''
        Define the training step for the model.

        Parameters:
            batch: Batch of data.
            batch_idx: Index of the current batch.

        Returns:
            torch.Tensor: The computed loss for training.
        '''
        if self.loss_type == 'type 1':
            loss = self.type_1_loss(batch)
        elif self.loss_type == 'type 2':
            loss = self.type_2_loss(batch) 
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("my_lr", cur_lr, prog_bar=True, on_step=True, sync_dist=True)
        return 100*loss
    
    def validation_step(self, batch, batch_idx):
        '''
        Define the validation step for the model.

        Parameters:
            batch: Batch of data.
            batch_idx: Index of the current batch.

        Returns:
            torch.Tensor: The computed loss for validation.
        '''
        self.epochs += 1
        torch.set_grad_enabled(True)
        
        if self.loss_type == 'type 1':
            Y_path_loss, Y_terminal_loss, dY_terminal_loss, loss = self.type_1_loss(batch, verbose=True)
        elif self.loss_type == 'type 2':
            Y_path_loss, Y_terminal_loss, dY_terminal_loss, loss = self.type_2_loss(batch, verbose=True) 
            
        print(self.epochs)
        print('LOSS:', loss.item())
        print('PATH LOSS:', Y_path_loss.mean().item())
        print('TERMINAL LOSS:', Y_terminal_loss.mean().item())
        print('TERMINAL GRAD LOSS:', dY_terminal_loss.mean().item())
        
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("my_lr", cur_lr, prog_bar=True, on_step=True, sync_dist=True)
        self.trainer.optimizers[0].zero_grad()
        return 100*loss

    def test_step(self, batch, batch_idx):
        '''
        Define the testing step for the model.

        Parameters:
            batch: Batch of data.
            batch_idx: Index of the current batch.

        Returns:
            torch.Tensor: The computed loss for testing.
        '''
        torch.set_grad_enabled(True)
        X, time, increments, B = batch     
        X = X.float()  
        Y_star, Z = self.FBSDE.compute_YZ(X, time, self.model_trajs)
        Y_0 = Y_star[:,0,:]
        Y = self.FBSDE.Y_not_run(Y_0, Z, increments, B)

        X_N_extra = self.boundary_args['reference'].view(1, X.size(2), -1).repeat(int(X.size(0)*.1), 1, 1)
        X_N = torch.cat((X[:,-1,:,:], X_N_extra), dim=0)
        t_N = torch.cat((time[:, -1], time[0:int(X.size(0)*.1), -1]), dim=0).view(-1,1)
        Y_star_N, Z_N = self.FBSDE.compute_YZ(X_N, t_N, self.model_trajs)

        loss, Y_path_loss, Y_terminal_loss, dY_terminal_loss = self.FBSDE.FBSDE_Loss( Y, Y_star, Y_star_N, Z_N, X_N, self.Boundary.Function)

        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("my_lr", cur_lr, prog_bar=True, on_step=True, sync_dist=True)
        self.trainer.optimizers[0].zero_grad()
        return loss
    
    def configure_optimizers(self):
        '''
        Configure the optimizer and learning rate scheduler.

        Returns:
            list: A list of optimizers and their corresponding schedulers.
        '''
        optimizer = optim.AdamW(self.parameters(), lr=self.initial_lr)
        sched = ReduceLROnPlateau(optimizer, factor=0.5, patience=15)
        scheduler = {"scheduler": sched, "monitor": "val_loss"}
        return [optimizer], scheduler
