import torch
from tqdm import tqdm
from torchani.utils import _get_derivatives_not_none as derivative
import copy
from torch.autograd import Variable


class FBSDE():

    def __init__(self, 
    grad = True,
    alpha = 1,
    beta = 1
    ) -> None:
        self.grad = grad
        self.alpha = alpha
        self.beta = beta

    def compute_YZ(self, X, t, model):

        X = Variable(X, requires_grad = True)
        Y = model(X, t)
        Z = derivative(X, Y, retain_graph = True, create_graph = True)
        return Y, Z

    def Y_step(self, Y_t, Z_t, wiener_increment_t, B):
        """
        Runs a single step Euler-Maruyama integration starting in the current state in the non-driven case
        """
        temp = torch.einsum('ijkl,ijl->ijk', B, wiener_increment_t)
        increment = torch.einsum('ijk,ijk->i', Z_t, temp).view(-1,Y_t.size(1))
        return Y_t + increment
    
    def Y_run(self, Y_0, Z, wiener_increments, B):

        steps = wiener_increments.size(1)
        trajs = Y_0.view(Y_0.size(0), 1, Y_0.size(1))
        Y_t = Y_0
        wiener_increment_t = wiener_increments[:, 0, :, :]
        Z_t = Z[:, 0, :, :]

        for t in tqdm(range(1,steps)):
            Y_t = self.Y_step(Y_t, Z_t, wiener_increment_t, B)
            trajs = torch.cat((trajs, Y_t.view(Y_t.size(0), 1, Y_t.size(1))), dim = 1)
            wiener_increment_t = wiener_increments[:, t, :, :]
            Z_t = Z[:, t, :, :]

        return trajs

    def Y_path_loss(self, Y, Y_star):
        path_loss_Y = ((Y - Y_star)**2).sum(dim = 2)
        return path_loss_Y.mean(dim = 1).mean(dim = 0)

    def Y_terminal_loss(self, Y_star_N, X_N, boundary_func):
        terminal_loss_Y = ((Y_star_N - boundary_func(X_N))**2).sum(dim = 1)
        return terminal_loss_Y.mean(dim = 0)
    
    def dY_terminal_loss(self, Z_N, X_N, boundary_func):
        
        if self.grad == True:
            X_N = Variable(X_N, requires_grad = True)
            Z_N_star = derivative(X_N, boundary_func(X_N), retain_graph = True, create_graph = True)
            terminal_loss_Z = ((Z_N - Z_N_star)**2).sum(dim = 2).sum(dim = 1)
            return terminal_loss_Z.mean(dim = 0)
        else:
            terminal_loss_Z = ((Z_N)**2).sum(dim = 2).sum(dim = 1)
            return terminal_loss_Z.mean(dim = 0)

    def FBSDE_Loss(self, Y, Y_star, Y_star_N, Z_N, X_N, boundary_func):
        
        Y_path_loss = self.Y_path_loss(Y, Y_star)
        Y_terminal_loss = self.Y_terminal_loss(Y_star_N, X_N, boundary_func)
        dY_terminal_loss = self.dY_terminal_loss(Z_N, X_N, boundary_func)
        
        Loss = Y_path_loss + self.alpha * Y_terminal_loss + self.beta * dY_terminal_loss

        return 100*Loss, Y_path_loss, Y_terminal_loss, dY_terminal_loss
    