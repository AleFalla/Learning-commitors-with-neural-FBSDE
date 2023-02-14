import torch
import numpy as np

class Boundary_functions():
    """
    a set of functions to approximate the indicator function to different degrees
    """
    def __init__(self,
    reference, 
    tolerance, 
    slope, 
    mask,
    keyword = 'Indicator'
    ) -> None:

        self.reference = reference
        self.tolerance = tolerance
        self.slope = slope
        self.mask = mask
        self.keyword = keyword
            
    def Sigmoid(self, state):
        """
        Implementation of sigmoig function, if slope is too small gives nan gradients
        """
        x = (((self.mask*(state-self.reference))**2).sum(dim = 1)**0.5).view(-1,1)
        return 1/(1+torch.exp((x-self.tolerance)/self.slope))

    def Indicator_function(self, state):
        """
        Implementation of actual indicator function, to be used asking gradients to be zero (non differentiable on a zero measure set)
        """
        x = torch.norm(self.mask*(state - self.reference),  dim = 2)
        y = x - self.tolerance
        where_0 = y>0
        where_1 = y<=0
        y[where_0] = 0.*y[where_0]
        y[where_1] = y[where_1]/y[where_1]
        y = y.prod(dim = 1)
        return y.view(-1,1)

    def Bump_function(self, state):
        """
        Implementation of Bump function, sort of works in all cases
        """
        x = 2*torch.norm(self.mask.cuda()*(state - self.reference.cuda()),  dim = 2)/self.tolerance
        func = (x)**2
        func = 1-func
        func = -self.slope/func
        func = (np.exp(self.slope))*torch.exp(func)
        func[x>=self.tolerance] = 0.*func[x>=self.tolerance]
        func = func.prod(dim = 1)
        func = func.view(-1,1)
        return func
    
    def Function(self, state):

        if self.keyword == 'Indicator':
            return self.Indicator_function(state)

        if self.keyword == 'Bump':
            return self.Bump_function(state)

        if self.keyword == 'Sigmoid':
            return self.Sigmoid(state)