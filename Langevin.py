import torch
from tqdm import tqdm
from torchani.utils import _get_derivatives_not_none as derivative
import copy
from torch.autograd import Variable

class Langevin_Dyn():

    def __init__(self, 
    Time_duration,
    dt,
    temperature,
    initial_state,
    masses,
    drag,
    potential,
    potential_args,
    device = 'cuda'
    ) -> None:
        
        """
        Class to do Langevin dynamics and generate path data

        Time_duration: [float] total time duration
        dt: [float] time interval for Euler-Maruyama integration
        temperature: [float tensor] temperature (same for all of the paths) or temperatures (different per path)
        initial_state: [float tensor] an initial state to get the dimensionality right and for path data generation, should be [paths, particles, ph_space]
        masses: [float tensor] mass (same for all of the particles) or masses (different per particle)
        drag: [float] drag coefficient
        potential: [func] potential to drive dynamics
        potential_args: [dict] extra arguments for the potential
        device: [str] device either 'cuda' or 'cpu'
        """

        self.steps = int(Time_duration/dt)
        self.dt = dt
        self.initial_state = initial_state
        self.dimensionality = int(initial_state.size(2)/2)
        self.particle_number = int(initial_state.size(1))
        self.paths_number = initial_state.size(0)
        self.potential = potential
        self.potential_args = potential_args
        self.device = device

        # check the length of masses list
        # if one then it's the same for all particles and can be just a float
        if len(masses) == 1:
            self.masses = masses.view(-1)
        
        # if bigger than 1 then it's different for each particle
        elif len(masses) > 1:
            self.masses = masses
        
        # if the number of masses and the number of particles extrapolated by the initial state is different, raise an error
        elif len(masses) != int(initial_state.size(1)/2):
            raise TypeError("you used a masses vector, but it has a different size wrt the chosen system")

        # check the length of temperature list
        # if one then consider the same for all trajectories, hence repeat it
        if len(temperature) == 1:
            self.temperature = temperature.view(-1)

        # if more than one, then turn it into tensor and that's it
        elif len(temperature)>1:
            self.temperature = temperature
        
        # if the size of temperatures is different wrt the number of paths as extrapolated from the initial state then raise an error
        elif len(temperature)!=int(initial_state.size(0)):
            raise TypeError("you used a temperature vector, but it has a different size wrt the batch of simulations")

        # repeat the drag coefficient dimensionality*particles times
        self.drag = drag

        self.diffusion_coeff = None

    def compute_drifts(self, current_state):
        """
        Computes the drift component A in the Langevin SDE dX = Adt + BdW
        """
        #separate velocities and positions
        velocities = current_state[:, :, self.dimensionality::]
        positions = current_state[:, :, 0:self.dimensionality]

        #compute deterministic forces
        positions = Variable(positions, requires_grad = True)
        U = self.potential(positions, **self.potential_args)
        forces = - derivative(positions, U, retain_graph = True, create_graph = True)

        #get the drift components that will go in the update
        position_drift = velocities
        velocity_drift_drag = - torch.einsum('ijk,j->ijk', velocities, self.drag/self.masses) # -(gamma/m_j)*v_jk where j runs on particles and k on dimensionality (e.g. xyz)
        velocity_drift_potential = torch.einsum('ijk,j->ijk', forces, 1/self.masses) # (1/m_j)*(-d(U(x))/dx_jk) where j runs on particles and k on dimensionality (e.g. xyz)
        velocity_drift = velocity_drift_drag + velocity_drift_potential
        A = torch.cat((position_drift, velocity_drift), dim = 2)

        return A
    

    def compute_diffusions(self, current_state):
        """
        Computes the diffusion component B for the Langevni SDE dX = Adt + BdW
        """
        
        # here we have the outer product of 2*gamma*T_i and 1/m_j where i runs on paths and j runs on particles
        Gammas = torch.einsum('i,j->ij', ((2*self.drag*self.temperature)**0.5), 1/self.masses)

        #here I generate the generic embedding for the diffusion matrices, there will be one per path per particle essentially
        embedding = torch.eye(current_state.size(2), device = self.device).view(1,1,current_state.size(2), current_state.size(2))
        embedding = embedding.repeat(current_state.size(0), current_state.size(1), 1, 1)
        embedding[:, :, 0:self.dimensionality, 0:self.dimensionality] = 0.

        #actual computation of diffusion matrices
        Diffusions = torch.einsum('ijkl, ij -> ijkl', embedding, Gammas)

        return Diffusions


    def compute_driven_drifts(self, current_state, committor_model, time):
        """
        Computes the drift component A in the Langevin SDE dX = Adt + BdW considering also the optimal driving -grad(log(q)) with q committor
        """
        
        #separate velocities and positions
        velocities = current_state[:, :, self.dimensionality::]
        positions = current_state[:, :, 0:self.dimensionality]

        #compute deterministic forces
        positions = Variable(positions, requires_grad = True)
        U = self.potential(positions, **self.potential_args)
        forces = - derivative(positions, U, retain_graph = True, create_graph = True)

        #compute the control part of the drift, that is to say 
        current_state = Variable(current_state, requires_grad = True)
        hitting_prob = committor_model(current_state, time)
        grad_log_q = derivative(current_state, torch.log(hitting_prob))
        
        #get the driving term
        B = self.compute_diffusions(current_state)
        B_2 = torch.einsum('ijkm, ijml -> ijkl', B, torch.transpose(B, 2, 3))
        driving = torch.einsum('ijkl, ijl -> ijk', B_2, grad_log_q)
        
        #compute the total driving for positions and velocities separately
        position_drift_driving = driving[:, :, 0:self.dimensionality]
        velocity_drift_driving = driving[:, :, self.dimensionality::]
        position_drift = velocities + position_drift_driving
        velocity_drift_drag = - torch.einsum('ijk,j->ijk', velocities, self.drag/self.masses) # -(gamma/m_j)*v_jk where j runs on particles and k on dimensionality (e.g. xyz)
        velocity_drift_potential = torch.einsum('ijk,j->ijk', forces, 1/self.masses) # (1/m_j)*(-d(U(x))/dx_jk) where j runs on particles and k on dimensionality (e.g. xyz)
        velocity_drift = velocity_drift_drag + velocity_drift_potential + velocity_drift_driving
        
        #stick it back together
        A = torch.cat((position_drift, velocity_drift), dim = 2)

        return A
    
    
    def Dyn_step(self, current_state, wiener_increment):
        """
        Runs a single step Euler-Maruyama integration starting in the current state in the non-driven case
        """
        A = self.compute_drifts(current_state)
        B = self.compute_diffusions(current_state)
        increment = A * self.dt + torch.einsum('ijkl, ijl -> ijk', B, wiener_increment)
        return current_state + increment, B
    
    def Dyn_step_driven(self, current_state, committor_model, time, wiener_increment):
        """
        Runs a single step Euler-Maruyama integration starting in the current state in the driven case using the specified model for hitting probability calculation
        """
        A = self.compute_driven_drifts(current_state, committor_model, time)
        B = self.compute_diffusions(current_state)
        increment = A * self.dt + torch.einsum('ijkl, ijl -> ijk', B, wiener_increment)
        return current_state + increment, B
    
    def Dyn_run(self, current_state):
        """
        Runs a full Langevin dynamics for the number of steps specified in the initialization of the class,
        Starts in the current state and returns trajectories and time axis
        """
        # Here I decided for a format (batch_size, time, particles, ph_space coordinates)
        trajs = current_state.view(current_state.size(0), 1, current_state.size(1), current_state.size(2))
        time_axis = [0]
        wiener_increment = torch.zeros_like(current_state, device = self.device)
        wiener_increment = wiener_increment.float()
        wiener_increments = (wiener_increment.view((current_state.size(0), 1, current_state.size(1), current_state.size(2))),)

        # time iteration
        for t in tqdm(range(1,self.steps+1)):
            current_state, B = self.Dyn_step(current_state, wiener_increment)
            trajs = torch.cat((trajs, current_state.view(current_state.size(0), 1, current_state.size(1), current_state.size(2))), dim = 1)
            time_axis.append(t*self.dt)
            wiener_increment = torch.randn_like(current_state, device = self.device)*(self.dt**0.5)
            wiener_increment = wiener_increment.float()
            wiener_increments = wiener_increments + (wiener_increment.view((current_state.size(0), 1, current_state.size(1), current_state.size(2))),)
            
        
        #turn to tensors time and wiener increments
        time_axis = torch.tensor(time_axis, device = self.device).view(1,-1,1).repeat(current_state.size(0), 1, 1)
        wiener_increments = torch.cat(wiener_increments, dim = 1)
        
        return trajs, time_axis, wiener_increments, B
    
    def Dyn_run_driven(self, current_state, committor_model):
        """
        Runs a full Langevin dynamics for the number of steps specified in the initialization of the class, driving the dynamics with the specified model for the committor
        Starts in the current state and returns trajectories and time axis
        """
        # Here I decided for a format (batch_size, time, system coordinates)
        trajs = current_state.view(current_state.size(0), 1, current_state.size(1), current_state.size(2))
        
        time_axis = [0]
        t = 0
        time = torch.tensor(t*self.dt, device = self.device).repeat(current_state.size(0), 1)

        wiener_increment = torch.zeros_like(current_state, device = self.device)
        wiener_increment = wiener_increment.float()

        for t in tqdm(range(1,self.steps+1)):
            current_state, B = self.Dyn_step_driven(current_state, committor_model, time, wiener_increment)
            trajs = torch.cat((trajs, current_state.view(current_state.size(0), 1, current_state.size(1), current_state.size(2))), dim = 1)
            time_axis.append(t*self.dt)
            time = torch.tensor(t*self.dt, device = self.device).repeat(current_state.size(0), 1)
            wiener_increment = torch.randn_like(current_state, device = self.device)*(self.dt**0.5)
            wiener_increment = wiener_increment.float()
            
        time_axis = torch.tensor(time_axis, device = self.device)

        return trajs, time_axis, B
    
    def Data_generation(self):

        """
        Generates trajectories based on the args given in initialization
        Returns trajectories, time, wiener increments and diffusion matrices
        """
        trajs, time_axis, wiener_increments, B = self.Dyn_run(self.initial_state)
        return trajs, time_axis, wiener_increments, B

            



        
    
    
    