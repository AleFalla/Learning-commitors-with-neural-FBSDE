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


def Create_uniform_initial(pow2_samples, initial_state, final_state, box_size, temperature, device = 'cuda'):
    """
    Gives uniformly random initial conditions centered in initial and final state of dimension 2**pow2_samples using Latin Hypercube method for uniform sampling.
    Positions are rescaled to the box size and velocities are sampled from boltzmann distribution

    Arguments:
        pow2_samples: [int] defines the lenght of the dataset in base 2, e.g. N = 2**pow_samples
        initial_state: [Tensor, (particles, ph_space_dim)] initial state (velocities to zero) for the brownian bridge as reference for properly centered sampling
        final_state: [Tensor, (particles, ph_space_dim)] target state (velocities to zero) for the brownian bridge as reference for properly centered sampling
        box_size: [float] width of the box defining the problem (positions). Velocities will be sampled from Boltzmann pdf.
        temperature: [float] temperature of choice for initial condition velocities sampling
        
    Returns:
        initial conditions samples [Tensor, (2**pow2_samples, particles, ph_space_dim)]
    """

    # Get dimensionality of the phase space from one of the states
    particles = final_state.size(0)
    ph_space_dim = final_state.size(1)
    sampler = qmc.LatinHypercube(d = ph_space_dim*particles)
    samples = torch.tensor(sampler.random(n = int(2**pow2_samples)), device = device)
    samples = 2*(samples - 0.5)
    samples = samples.view(samples.size(0), particles, ph_space_dim)
    
    # these are boxes but we want spheres so we normalize and multiply by random_norm**0.5 
    # first for positions (then rescale to the box size)
    samples[:, :, 0:int(ph_space_dim*0.5)] = samples[:, :, 0:int(ph_space_dim*0.5)]/samples[:, :, 0:int(ph_space_dim*0.5)].norm(dim = 2).view(samples.size(0), particles, 1)
    norm_sampler = qmc.LatinHypercube(d = particles)
    norm_samples = torch.tensor(norm_sampler.random(n = int(2**pow2_samples)), device = device).view(int(2**pow2_samples), particles, 1)
    samples[:, :, 0:int(ph_space_dim*0.5)] = samples[:, :, 0:int(ph_space_dim*0.5)]*(norm_samples**0.5)
    samples[:, :, 0:int(ph_space_dim*0.5)] = samples[:, :, 0:int(ph_space_dim*0.5)] * box_size
    
    # then for velocities, but this time the norms are taken from a maxwell distribution
    samples[:, :, int(ph_space_dim*0.5)::] = samples[:, :, int(ph_space_dim*0.5)::]/samples[:, :, int(ph_space_dim*0.5)::].norm(dim = 2).view(samples.size(0), particles, 1)
    norm_samples = torch.tensor(maxwell.rvs(size = int(2**pow2_samples*particles), scale = temperature), device = device).view(int(2**pow2_samples), particles, 1)
    samples[:, :, int(ph_space_dim*0.5)::] = samples[:, :, int(ph_space_dim*0.5)::]*(norm_samples**0.5)
    
    # recenter around initial and final states
    sample_initial = samples[0:int(2**(pow2_samples-1))] + initial_state
    sample_final = samples[int(2**(pow2_samples-1))::] + final_state

    # and stick them back together
    initial_conditions_sample = torch.cat((sample_initial, sample_final), dim = 0)

    return initial_conditions_sample



