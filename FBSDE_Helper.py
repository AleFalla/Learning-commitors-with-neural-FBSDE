import torch
from torchani.utils import _get_derivatives_not_none as derivative

class FBSDE:
    """
    Class implementing the Forward-Backward Stochastic Differential Equation (FBSDE) solver.

    Parameters:
        grad (bool): Whether to use gradient information in loss computation (default is True).
        alpha (float): Weight for the terminal loss term (default is 1).
        beta (float): Weight for the derivative terminal loss term (default is 1).
    """

    def __init__(self, grad=True, alpha=1, beta=1):
        self.grad = grad
        self.alpha = alpha
        self.beta = beta

    def compute_YZ(self, X, t, model, graph=True):
        """
        Compute the model output (Y) and its derivatives (Z) with respect to the input (X) at time (t).

        Parameters:
            X (torch.Tensor): Input tensor of shape (batch_size, p).
            t (float): Time value.
            model: The PyTorch model for computing Y and its derivatives.
            graph (bool): Whether to create a computation graph (default is True).

        Returns:
            tuple: A tuple containing the model output (Y) and its derivatives (Z).
        """
        X.requires_grad_()
        Y = model(X, t)
        Z = derivative(X, Y, retain_graph=True, create_graph=graph)
        return Y, Z

    def compute_stepwise_Ys(self, Y, Z, wiener_increments, B):
        """
        Compute the stepwise values of Y.

        Parameters:
            Y (torch.Tensor): Tensor of shape (batch_size, time_steps, p) representing Y values.
            Z (torch.Tensor): Tensor of shape (batch_size, time_steps, particle, x12) representing Z values.
            wiener_increments (torch.Tensor): Tensor of shape (batch_size, time_steps, particle, x12) representing Wiener increments.
            B (float): Scaling factor for the increments.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, time_steps, p) representing stepwise Y values.
        """
        increments = B * torch.einsum('ijkl, ijkl -> ijk', Z, wiener_increments).sum(-1)
        return Y + increments.view(Y.size(0), Y.size(1), Y.size(2))

    def OLD_Y_loss(self, Y, Y_incr):
        """
        Compute the loss based on the difference between Y and the incremental Y values.

        Parameters:
            Y (torch.Tensor): Tensor of shape (batch_size, time_steps, p) representing Y values.
            Y_incr (torch.Tensor): Tensor of shape (batch_size, time_steps, p) representing the incremental Y values.

        Returns:
            torch.Tensor: Tensor of shape (batch_size,) representing the loss.
        """
        delta = Y[:, 1:, :] - Y_incr[:, :-1, :]
        return (delta**2).sum(-1).mean(-1)

    def Y_step(self, Y_t, Z_t, wiener_increment_t, B):
        """
        Perform a single step Euler-Maruyama integration starting from the current state.

        Parameters:
            Y_t (torch.Tensor): Tensor of shape (batch_size, p) representing Y values at time t.
            Z_t (torch.Tensor): Tensor of shape (batch_size, particle, x12) representing Z values at time t.
            wiener_increment_t (torch.Tensor): Tensor of shape (batch_size, particle, x12) representing Wiener increments at time t.
            B (float): Scaling factor for the increments.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, p) representing the updated Y values.
        """
        scalar = torch.einsum('ijk, ijk -> ij', Z_t, wiener_increment_t).sum(-1).view(-1, 1)
        increment = B * scalar
        return Y_t + increment

    def Y_run(self, Y_0, Z, wiener_increments, B):
        """
        Perform the Euler-Maruyama integration for Y values over time.

        Parameters:
            Y_0 (torch.Tensor): Tensor of shape (batch_size, p) representing the initial Y values.
            Z (torch.Tensor): Tensor of shape (batch_size, time_steps, particle, x12) representing Z values.
            wiener_increments (torch.Tensor): Tensor of shape (batch_size, time_steps, particle, x12) representing Wiener increments.
            B (float): Scaling factor for the increments.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, time_steps, p) representing the Y trajectories.
        """
        steps = wiener_increments.size(1)
        trajs = Y_0.view(Y_0.size(0), 1, Y_0.size(1))
        Y_t = Y_0
        wiener_increment_t = wiener_increments[:, 0, :, :]
        Z_t = Z[:, 0, :, :]

        for t in range(1, steps):
            Y_t = self.Y_step(Y_t, Z_t, wiener_increment_t, B)
            trajs = torch.cat((trajs, Y_t.view(Y_t.size(0), 1, Y_t.size(1))), dim=1)
            wiener_increment_t = wiener_increments[:, t, :, :]
            Z_t = Z[:, t, :, :]

        return trajs

    def Y_path_loss(self, Y, Y_star):
        """
        Compute the loss based on the difference between Y and the target Y values.

        Parameters:
            Y (torch.Tensor): Tensor of shape (batch_size, time_steps, p) representing Y values.
            Y_star (torch.Tensor): Tensor of shape (batch_size, time_steps, p) representing the target Y values.

        Returns:
            torch.Tensor: Tensor of shape (batch_size,) representing the loss.
        """
        path_loss_Y = ((Y - Y_star)**2).mean(dim=-1)
        return path_loss_Y.mean(dim=-1)

    def Y_terminal_loss(self, Y_star_N, X_N, boundary_func):
        """
        Compute the terminal loss based on the difference between Y_star_N and the boundary function.

        Parameters:
            Y_star_N (torch.Tensor): Tensor of shape (batch_size, p) representing Y_star_N values.
            X_N (torch.Tensor): Tensor of shape (batch_size, p) representing the input X values at the terminal time step.
            boundary_func: The boundary function.

        Returns:
            torch.Tensor: Tensor of shape (batch_size,) representing the terminal loss.
        """
        terminal_loss_Y = ((Y_star_N - boundary_func(X_N))**2).sum(dim=1)
        return terminal_loss_Y

    def dY_terminal_loss(self, Z_N, X_N, boundary_func):
        """
        Compute the derivative terminal loss based on the difference between Z_N and the derivative of the boundary function.

        Parameters:
            Z_N (torch.Tensor): Tensor of shape (batch_size, particle, x12) representing Z_N values.
            X_N (torch.Tensor): Tensor of shape (batch_size, p) representing the input X values at the terminal time step.
            boundary_func: The boundary function.

        Returns:
            torch.Tensor: Tensor of shape (batch_size,) representing the derivative terminal loss.
        """
        if self.grad:
            X_N.requires_grad_()
            Z_N_star = derivative(X_N, boundary_func(X_N), retain_graph=True)
            terminal_loss_Z = ((Z_N - Z_N_star).norm(dim=-1)**2).mean(dim=-1)
            # If needed, mask values for specific conditions using the code commented below
            # mask = torch.ones_like(terminal_loss_Z).to(X_N.device)
            # mask[Z_N_star.norm(dim=-1).view(-1) >= 1] = 1e-3
            # terminal_loss_Z = mask * terminal_loss_Z
        else:
            terminal_loss_Z = (Z_N**2).sum(dim=-1).sum(dim=-1)
        return terminal_loss_Z

    def FBSDE_Loss(self, Y, Y_star, Y_star_N, Z_N, X_N, boundary_func):
        """
        Compute the FBSDE loss.

        Parameters:
            Y (torch.Tensor): Tensor of shape (batch_size, time_steps, p) representing Y values.
            Y_star (torch.Tensor): Tensor of shape (batch_size, time_steps, p) representing the target Y values.
            Y_star_N (torch.Tensor): Tensor of shape (batch_size, p) representing Y_star_N values.
            Z_N (torch.Tensor): Tensor of shape (batch_size, particle, x12) representing Z_N values.
            X_N (torch.Tensor): Tensor of shape (batch_size, p) representing the input X values at the terminal time step.
            boundary_func: The boundary function.

        Returns:
            tuple: A tuple containing the total loss, the path loss, the terminal Y loss, and the derivative terminal loss.
        """
        Y_path_loss = self.Y_path_loss(Y, Y_star)
        Y_terminal_loss = self.Y_terminal_loss(Y_star_N, X_N, boundary_func)
        dY_terminal_loss = self.dY_terminal_loss(Z_N, X_N, boundary_func)

        Loss = Y_path_loss + self.alpha * Y_terminal_loss + self.beta * dY_terminal_loss

        return Loss, Y_path_loss, Y_terminal_loss, dY_terminal_loss



# import torch
# from tqdm import tqdm
# from torchani.utils import _get_derivatives_not_none as derivative
# import copy
# from torch.autograd import Variable


# class FBSDE():

#     def __init__(self, 
#     grad = True,
#     alpha = 1,
#     beta = 1
#     ) -> None:
#         self.grad = grad
#         self.alpha = alpha
#         self.beta = beta

#     def compute_YZ(self, X, t, model, graph = True):

#         X.requires_grad_()
#         Y = model(X, t)
#         Z = derivative(X, Y, retain_graph = True, create_graph = graph)
#         return Y, Z

#     def compute_stepwise_Ys(self, Y, Z, wiener_increments, B):
#         # Y = batch, time, p; Z = batch, time, particle, x12
#         increments = B * torch.einsum('ijkl, ijkl -> ijk', Z, wiener_increments).sum(-1)
        
#         return Y + increments.view(Y.size(0) ,Y.size(1), Y.size(2))
    
#     def OLD_Y_loss(self, Y, Y_incr):
#         delta = Y[:,1:,:] - Y_incr[:,:-1,:]
#         return (delta**2).sum(-1).mean(-1)
    
#     def Y_step(self, Y_t, Z_t, wiener_increment_t, B):
#         """
#         Runs a single step Euler-Maruyama integration starting in the current state in the non-driven case
#         """
#         scalar = torch.einsum('ijk, ijk -> ij', Z_t, wiener_increment_t).sum(-1).view(-1,1)
#         increment = B*scalar
#         return Y_t + increment
    
#     def Y_run(self, Y_0, Z, wiener_increments, B):
        
#         steps = wiener_increments.size(1)
#         trajs = Y_0.view(Y_0.size(0), 1, Y_0.size(1))
#         Y_t = Y_0
#         wiener_increment_t = wiener_increments[:, 0, :, :]
#         Z_t = Z[:, 0, :, :]

#         for t in (range(1,steps)):
#             Y_t = self.Y_step(Y_t, Z_t, wiener_increment_t, B)
#             trajs = torch.cat((trajs, Y_t.view(Y_t.size(0), 1, Y_t.size(1))), dim = 1)
#             wiener_increment_t = wiener_increments[:, t, :, :]
#             Z_t = Z[:, t, :, :]

#         return trajs
    
#     def Y_path_loss(self, Y, Y_star):
#         path_loss_Y = ((Y - Y_star)**2).mean(dim = -1)
#         return path_loss_Y.mean(dim = -1)

#     def Y_terminal_loss(self, Y_star_N, X_N, boundary_func):
#         #print(boundary_func(X_N))
#         #print(Y_star_N.size(), boundary_func(X_N).size())
#         terminal_loss_Y = ((Y_star_N - boundary_func(X_N))**2).sum(dim = 1)
#         return terminal_loss_Y
    
#     def dY_terminal_loss(self, Z_N, X_N, boundary_func):
        
#         if self.grad == True:
#             X_N.requires_grad_() 
#             Z_N_star = derivative(X_N, boundary_func(X_N), retain_graph = True)#, create_graph = True)
#             terminal_loss_Z = ((Z_N - Z_N_star).norm(dim = -1)**2).mean(dim = -1)
#             # mask = torch.ones_like(terminal_loss_Z).to(X_N.device)
#             # mask[Z_N_star.norm(dim = -1).view(-1)>=1] = 1e-3
#             # terminal_loss_Z = mask*terminal_loss_Z
#             return terminal_loss_Z
        
#         else:
#             terminal_loss_Z = (Z_N**2).sum(dim = -1).sum(dim = -1)
#             return terminal_loss_Z

#     def FBSDE_Loss(self, Y, Y_star, Y_star_N, Z_N, X_N, boundary_func):
        
#         Y_path_loss = self.Y_path_loss(Y, Y_star)
#         Y_terminal_loss = self.Y_terminal_loss(Y_star_N, X_N, boundary_func)
#         dY_terminal_loss = self.dY_terminal_loss(Z_N, X_N, boundary_func)
        
#         Loss = Y_path_loss + self.alpha * Y_terminal_loss + self.beta * dY_terminal_loss

#         return Loss, Y_path_loss, Y_terminal_loss, dY_terminal_loss
    