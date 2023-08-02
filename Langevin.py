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
    potential,
    drag,
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

        self.drag = drag
        self.Gamma = ((2*self.temperature/self.drag)**0.5).view(-1,1)

    def compute_drifts(self, current_state):
        """
        Computes the drift component A in the Langevin SDE dX = Adt + BdW
        In this case A*dt = -gradU(x)*dt/gamma 
        """
        
        #separate velocities and positions
        positions = current_state

        #compute deterministic forces
        positions.requires_grad_() #= Variable(positions, requires_grad = True)
        
        U = self.potential(positions, **self.potential_args)
        
        forces = - derivative(positions, U, retain_graph = True, create_graph = True)

        #get the drift components that will go in the update
        A = forces/self.drag

        return A


    def compute_driven_drifts(self, current_state, committor_model, time):
        """
        Computes the drift component A in the Langevin SDE dX = Adt + BdW considering also the optimal driving -grad(log(q)) with q committor
        """

        #compute deterministic forces
        positions = current_state
        positions.requires_grad_()# = Variable(current_state, requires_grad = True)
        U = self.potential(positions, **self.potential_args)
        forces = - derivative(positions, U, retain_graph = True, create_graph = True)

        #compute the control part of the drift, that is to say 
        positions = current_state
        positions.requires_grad_()# = Variable(current_state, requires_grad = True)
        hitting_prob = committor_model(positions, time)
        grad_log_q = derivative(positions, torch.log(hitting_prob))
        driving = (self.Gamma**2)*grad_log_q
        
        #get the drift components that will go in the update
        A = forces/self.drag + driving

        return A
    
    def Dyn_step(self, current_state, wiener_increment):
        """
        Runs a single step Euler-Maruyama integration starting in the current state in the non-driven case
        """
        A = self.compute_drifts(current_state)
        increment = A * self.dt + self.Gamma * wiener_increment
        
        return current_state + increment
    
    def Dyn_step_driven(self, current_state, committor_model, time, wiener_increment):
        """
        Runs a single step Euler-Maruyama integration starting in the current state in the driven case using the specified model for hitting probability calculation
        """
        A = self.compute_driven_drifts(current_state, committor_model, time)
        increment = A * self.dt + self.Gamma * wiener_increment
        
        return current_state + increment
    
    def Dyn_run(self, current_state):
        """
        Runs a full Langevin dynamics for the number of steps specified in the initialization of the class,
        Starts in the current state and returns trajectories and time axis
        """
        # Here I decided for a format (batch_size, time, particles, ph_space coordinates)
        trajs = current_state.view(current_state.size(0), 1, current_state.size(1), current_state.size(2))
        
        # I get the value for the drift term to initialize the array of drifts (to keep it in memory in case of PDE loss)
        A = self.compute_drifts(current_state)
        As = A.view(A.size(0), 1, A.size(1), A.size(2))
        
        # I initialize the time axis
        time_axis = [0]
        
        # I sample the first wiener_increment and initialize the vector of increments to keep memory for the integration of Y in the FBSDE method
        wiener_increment = (torch.randn_like(current_state)*(self.dt**0.5)).to(current_state.device)
        wiener_increment = wiener_increment.float()
        wiener_increments = (wiener_increment.view((current_state.size(0), 1, current_state.size(1), current_state.size(2))),)

        # at this point I have t=0, the current state as initial state, the sampled wiener increment and the drift term initialized based on the current state
        # then I go into the time iteration
        for t in tqdm(range(1,self.steps+1)):
            
            # first I update the current state X_{t-1} -> X_{t} of the system based on the values of wiener increment and drift (computed internally as a function of X_{t-1}) at time t-1  
            current_state = self.Dyn_step(current_state, wiener_increment)
            
            # now I compute the new drifts and append them
            A = self.compute_drifts(current_state)
            As = torch.cat((As, A.view(A.size(0), 1, A.size(1), A.size(2))), dim = 1)
            
            # append the new state to the trajectories tensor
            trajs = torch.cat((trajs, current_state.view(current_state.size(0), 1, current_state.size(1), current_state.size(2))), dim = 1)
            
            # append the updated time
            time_axis.append(t*self.dt)
            
            # finally I sample the new wiener increment dW_{t} and append it to the wiener increments vector
            wiener_increment = (torch.randn_like(current_state)*(self.dt**0.5)).to(current_state.device)
            wiener_increment = wiener_increment.float()
            wiener_increments = wiener_increments + (wiener_increment.view((current_state.size(0), 1, current_state.size(1), current_state.size(2))),)
            
            # at this point I have time t, state X_{t} and wiener increment dW_{t} which will be used in the next iteration to do the update t->t+1
            # consider that at every time we noe have (t, X_{t}, dW_{t}) and that is all we need
            
        
        # finally I turn the time axis into a tensor, repeated per each batch and with extra dummy dimension for each scalar value
        time_axis = (torch.tensor(time_axis).view(1,-1,1).repeat(current_state.size(0), 1, 1)).to(current_state.device)
        
        # I make the wiener increments tuple into a tensor
        wiener_increments = torch.cat(wiener_increments, dim = 1)
        
        # for the B component the cases are two, either we are considering a different temperature per trajectory or all the same
        if self.Gamma.size(0) == 1:
            B = self.Gamma.repeat(current_state.size(0), 1)
        else:
            B = self.Gamma
        
        return trajs, time_axis, wiener_increments, As, B
    
    def Dyn_run_driven(self, current_state, committor_model):
        """
        Runs a full Langevin dynamics for the number of steps specified in the initialization of the class, driving the dynamics with the specified model for the committor
        Starts in the current state and returns trajectories and time axis
        """
        # Here I decided for a format (batch_size, time, particles, ph_space coordinates)
        trajs = current_state.view(current_state.size(0), 1, current_state.size(1), current_state.size(2))
        
        # I get the value for the drift term to initialize the array of drifts (to keep it in memory in case of PDE loss)
        A = self.compute_driven_drifts(current_state, committor_model, torch.tensor([0]).view(1,-1).repeat(current_state.size(0),1))
        As = A.view(A.size(0), 1, A.size(1), A.size(2))
        
        # I initialize the time axis
        time_axis = [0]
        
        # I sample the first wiener_increment and initialize the vector of increments to keep memory for the integration of Y in the FBSDE method
        wiener_increment = (torch.randn_like(current_state)*(self.dt**0.5)).to(current_state.device)
        wiener_increment = wiener_increment.float()
        wiener_increments = (wiener_increment.view((current_state.size(0), 1, current_state.size(1), current_state.size(2))),)

        # at this point I have t=0, the current state as initial state, the sampled wiener increment and the drift term initialized based on the current state
        # then I go into the time iteration
        for t in tqdm(range(1,self.steps+1)):
            
            # first I update the current state X_{t-1} -> X_{t} of the system based on the values of wiener increment and drift (computed internally as a function of X_{t-1}) at time t-1  
            current_state = self.Dyn_step_driven(current_state, committor_model, torch.tensor([(t-1)*self.dt]).view(1,-1).repeat(current_state.size(0),1), wiener_increment)
            
            # now I compute the new drifts and append them
            A = self.compute_driven_drifts(current_state, committor_model, torch.tensor([t*self.dt]).view(1,-1).repeat(current_state.size(0),1))
            As = torch.cat((As, A.view(A.size(0), 1, A.size(1), A.size(2))), dim = 1)
            
            # append the new state to the trajectories tensor
            trajs = torch.cat((trajs, current_state.view(current_state.size(0), 1, current_state.size(1), current_state.size(2))), dim = 1)
            
            # append the updated time
            time_axis.append(t*self.dt)
            
            # finally I sample the new wiener increment dW_{t} and append it to the wiener increments vector
            wiener_increment = (torch.randn_like(current_state)*(self.dt**0.5)).to(current_state.device)
            wiener_increment = wiener_increment.float()
            wiener_increments = wiener_increments + (wiener_increment.view((current_state.size(0), 1, current_state.size(1), current_state.size(2))),)
            
            # at this point I have time t, state X_{t} and wiener increment dW_{t} which will be used in the next iteration to do the update t->t+1
            # consider that at every time we noe have (t, X_{t}, dW_{t}) and that is all we need
            
        
        # finally I turn the time axis into a tensor, repeated per each batch and with extra dummy dimension for each scalar value
        time_axis = (torch.tensor(time_axis).view(1,-1,1).repeat(current_state.size(0), 1, 1)).to(current_state.device)
        
        # I make the wiener increments tuple into a tensor
        wiener_increments = torch.cat(wiener_increments, dim = 1)
        
        # for the B component the cases are two, either we are considering a different temperature per trajectory or all the same
        if self.Gamma.size(0) == 1:
            B = self.Gamma.repeat(current_state.size(0), 1)
        else:
            B = self.Gamma
        
        return trajs, time_axis, B
    
    def Data_generation(self):

        """
        Generates trajectories based on the args given in initialization
        Returns trajectories, time, wiener increments and diffusion matrices
        """
        trajs, time_axis, wiener_increments, A, B = self.Dyn_run(self.initial_state)
        return trajs, time_axis, wiener_increments, A, B

            



        
    
    
    