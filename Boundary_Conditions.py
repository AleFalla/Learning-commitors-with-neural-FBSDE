import torch
import numpy as np

class Boundary_functions:
    """
    A set of functions to approximate the indicator function to different degrees.
    """

    def __init__(self, reference, tolerance, slope, mask, keyword='Indicator'):
        """
        Initialize the Boundary_functions object.

        Parameters:
            reference (torch.Tensor): Reference tensor representing the reference point.
            tolerance (float): Tolerance value for the boundary function.
            slope (float): Slope value used in some boundary functions.
            mask (torch.Tensor): Mask tensor for the boundary function.
            keyword (str): Keyword specifying the type of boundary function (default is 'Indicator').
        """
        self.reference = reference
        self.tolerance = tolerance
        self.slope = slope
        self.mask = mask
        self.keyword = keyword
            
    def Sigmoid(self, state):
        """
        Implementation of the sigmoid function. If slope is too small, it gives NaN gradients.

        Parameters:
            state (torch.Tensor): Input tensor representing the state.

        Returns:
            torch.Tensor: Output tensor representing the sigmoid function's values.
        """
        x = torch.norm(self.mask * (state - self.reference), dim=-1)
        y = (x - self.tolerance) / self.slope
        return torch.exp(-y) / (torch.exp(-y) + 1)

    def Indicator_function(self, state):
        """
        Implementation of the actual indicator function. Used when gradients need to be zero (non-differentiable on a zero-measure set).

        Parameters:
            state (torch.Tensor): Input tensor representing the state.

        Returns:
            torch.Tensor: Output tensor representing the indicator function's values.
        """
        x = torch.norm(self.mask * (state - self.reference), dim=2)
        y = x - self.tolerance
        where_0 = y > 0
        where_1 = y <= 0
        y[where_0] = 0. * y[where_0] 
        y[where_1] = y[where_1] / y[where_1] 
        y = y.prod(dim=1).view(-1, 1)
        return y

    def Bump_function(self, state):
        """
        Implementation of the Bump function. Sort of works in all cases.

        Parameters:
            state (torch.Tensor): Input tensor representing the state.

        Returns:
            torch.Tensor: Output tensor representing the Bump function's values.
        """
        x = torch.norm(self.mask.to(state.device) * (state - self.reference.to(state.device)), dim=2) / self.tolerance
        func = torch.exp(self.slope - self.slope / (1 - x**2))
        mask = torch.ones_like(func).to(state.device)
        mask[x**2 > 1] = 0.
        func = func * mask
        func = func.prod(dim=1)
        func = func.view(-1, 1)
        return func
    
    def Function(self, state):
        """
        Wrapper function for different boundary functions.

        Parameters:
            state (torch.Tensor): Input tensor representing the state.

        Returns:
            torch.Tensor: Output tensor representing the boundary function's values.
        """
        if self.keyword == 'Indicator':
            return self.Indicator_function(state)

        if self.keyword == 'Bump':
            return self.Bump_function(state)

        if self.keyword == 'Sigmoid':
            return self.Sigmoid(state)
