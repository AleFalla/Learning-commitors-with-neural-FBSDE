from Data_Handler import Data_Handler
import torch
from Utilities import Create_uniform_initial
from Potentials import Mueller_Potential

# Check if CUDA is available, otherwise use CPU
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Parameters for Langevin dynamics and data generation
temperature = 7
box_size = 0.1
pow_2 = 13

# Define initial and final states (input as a 2D tensor)
initial_state = torch.tensor([[-0.6, 1.5]])

final_state = torch.tensor([[0.6, 0]])

# Move the states to the GPU if available
if device == 'cuda':
    initial_state = initial_state.cuda()
    final_state = final_state.cuda()

# Create uniform initial conditions for Langevin dynamics
initial_conds = Create_uniform_initial(pow_2, initial_state, final_state, box_size)

# Arguments for Langevin dynamics simulation
args = {
    'Time_duration': 1,
    'dt': 1/500,
    'temperature': torch.tensor([temperature], device=device),
    'initial_state': initial_conds,
    'drag': 10.,
    'potential': Mueller_Potential,
    'potential_args': {}
}

# Create Data_Handler object for data generation
Handler = Data_Handler(
    Langevyn_args=args,
    train_fraction=0.9,
    validation_fraction=0.1
)

# Generate and save the data
Handler.create_data_in_one_go(save_to_file=True)
