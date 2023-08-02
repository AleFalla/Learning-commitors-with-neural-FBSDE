import torch

def Mueller_Potential(x):
    #form A*exp(a*(x-xo)^2 + b*(x-xo)*(y-yo) + c*(y-yo)^2)
    A = [-200., -100., -170, 15]
    a = [-1.,-1.,-6.5,0.7]
    b = [0.,0.,11.,0.6]
    c = [-10.,-10.,-6.5,0.7]
    xo = [1., 0., -0.5, -1.]
    yo = [0., 0.5, 1.5, 1]
    energy = torch.zeros(x.size(0), 1).to(x.device)
    for i in range(0,4):
        ref = torch.tensor([xo[i],yo[i]]).to(x.device)
        delta = x-ref
        potential = A[i]*torch.exp(a[i]*(delta[:, :, 0]**2) + b[i]*delta[:,:,0]*delta[:,:,1] + c[i]*(delta[:, :, 1]**2))
        potential = potential.sum(dim = 1).view(-1, 1)
        energy+=potential
    return energy

