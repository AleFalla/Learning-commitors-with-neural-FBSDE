import torch
import torchani
from torchani.utils import _get_derivatives_not_none as derivative
from scipy.stats import qmc
import math
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import maxwell
from Langevin import Langevin_Dyn
import copy
import numpy as np
from torch.autograd import Variable

def Create_uniform_initial(pow2_samples, initial_state, final_state, box_size, interpolation_steps=1000):
    """
    Creates uniformly random initial conditions centered in initial and final states of dimension 2**pow2_samples using the Latin Hypercube method for uniform sampling.
    Positions are rescaled to the box size.

    Args:
        pow2_samples (int): Defines the length of the dataset in base 2, e.g., N = 2**pow_samples.
        initial_state (Tensor): Initial state for the Brownian bridge as reference for properly centered sampling.
        final_state (Tensor): Target state for the Brownian bridge as reference for properly centered sampling.
        box_size (float): Width of the box defining the problem (positions)
        interpolation_steps (int): Number of steps for linear interpolation between the initial and final states.

    Returns:
        Tensor: Initial condition samples of shape (2**pow2_samples, particles, d).
    """

    # Get dimensionality of the system from one of the states
    particles = final_state.size(0)
    d = final_state.size(1)

    # Prepare the sampler and sample
    sampler = qmc.LatinHypercube(d=d * particles)
    samples = torch.tensor(sampler.random(n=int(2**pow2_samples))).to(initial_state.device)
    samples = 2 * (samples - 0.5)
    samples = samples.view(samples.size(0), particles, d)

    # These are boxes, but we want spheres, so we normalize and multiply by random_norm**0.5
    # First for positions (then rescale to the box size)
    samples = samples / (samples.norm(dim=2).view(samples.size(0), particles, 1))
    norm_sampler = qmc.LatinHypercube(d=particles)
    norm_samples = (torch.tensor(norm_sampler.random(n=int(2**pow2_samples))).view(int(2**pow2_samples), particles, 1)).to(initial_state.device)
    samples = samples * (norm_samples**(1 / d))
    samples = samples * box_size

    # Recenter around initial and final states
    sample_initial = samples + initial_state
    sample_final = samples + final_state

    # Add linear interpolation between initial_state and final_state as initial conditions
    linear = (initial_state.view(1, particles, d),)
    for i in range(1, interpolation_steps + 1):
        c = torch.lerp(initial_state.view(1, particles, d), final_state.view(1, particles, d), i * (1 / interpolation_steps))
        linear += (c,)
    linear = torch.cat(linear, dim=0)

    print(sample_final.size(), linear.size())

    # Stick the initial condition samples back together
    initial_conditions_sample = torch.cat((sample_initial, sample_final, linear), dim=0)

    return initial_conditions_sample