from Data_Handler import Data_Handler
import torch
from Utilities import Create_uniform_initial

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def potential(x):
    c_0 = torch.tensor([-2,-2, -2,-2, -2,-2, -2,-2], device = device).view(1,8,-1).repeat(x.size(0), 1, 1)
    c_1 = torch.tensor([2,2, 2,2, 2,2, 2,2], device = device).view(1,8,-1).repeat(x.size(0), 1, 1)
    potential = (torch.exp(-((x/5).norm(dim = 2))**2)*((((x-c_0)**2).sum(dim = 2))*(((x-c_1)**2).sum(dim = 2)))).sum(dim = 1)
    potential.view(-1,1)
    return potential

temperature = 10
box_size = 4
pow_2 = 15

initial_state = torch.tensor([[2.,0.],
                              [2.,0.],
                              [2.,0.],
                              [2.,0.],
                              [2.,0.],
                              [2.,0.],
                              [2.,0.],
                              [2.,0.]], device = device)

final_state = torch.tensor([[-2.,0.],
                            [-2.,0.],
                            [-2.,0.],
                            [-2.,0.],
                            [-2.,0.],
                            [-2.,0.],
                            [-2.,0.],
                            [-2.,0.]], device = device)

initial_conds = Create_uniform_initial(
    pow_2, 
    initial_state, 
    final_state, 
    box_size, 
    temperature, 
    device = device
    )

args = {
    'Time_duration':1,
    'dt' : 1/1000,
    'temperature':torch.tensor([10.], device = device),
    'initial_state':initial_conds.view(int(2**pow_2), 8, -1),
    'masses':torch.tensor([1.], device = device),
    'drag':10.,
    'potential':potential,
    'potential_args':{},
    'device':device
}

Handler = Data_Handler(
    args, 
    device = device
)

Handler.create_data_in_one_go(save_to_file = True)
