# Functions for generating sample data NB: SEED IS SET WHEN LOADING THIS
#                                          MODULE!

import numpy as np
import scipy.sparse as sp
np.random.seed(12)

import warnings
#Comment this to turn on warnings
warnings.filterwarnings('ignore')


# Computes the energy of the system given L states
def ising_energies(states,L):
    """
    This function calculates the energies of the states in the nn Ising Hamiltonian
    """
    J=np.zeros((L,L),)
    for i in range(L):
        J[i,(i+1)%L]-=1.0
    # compute energies
    E = np.einsum('...i,ij,...j->...',states,J,states)

    return E


# system size
def generate_data(L, N):
    # create N random Ising states
    states = np.random.choice([-1, 1], size=(N,L))

    # calculate Ising energies
    energies = ising_energies(states,L)

    return states, energies


