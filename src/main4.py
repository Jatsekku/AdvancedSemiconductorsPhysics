
import numpy as np
import matplotlib.pyplot as plt


import functools
import itertools

from scipy import constants as c


# %% Material and physics constants

A = 5.43e-10 # length of the unit cell [m]
# Values of form factors in eV
form_factors = {3.0: np.array(-0.21*13.6059), 
               8.0: np.array(0.04*13.6059), 
               11.0:np.array(0.08*13.6059)} 


tau = 1/8 * np.array([1, 1, 1])

# Base vectors in the base of 2 pi / a
b1 = np.array([-1., 1., 1.])
b2 = np.array([1., -1., 1.])
b3 = np.array([1., 1., -1.])   
reciprocal_base= np.array([b1,b2,b3])

SCALE = (2 * np.pi / A)**2
KINETIC_CONSTANT = c.hbar**2 / (2 * c.m_e * c.e)

no_of_states = 7





# %% Defining the path for k vector in k-space
def path(start, end, n, endpoint):
    """
    Return n points along chosen way from start to end with or without end point
    """
    leg = []
    for i in range(3):
        leg.append(np.linspace(start[i], end[i], num=int(n), endpoint=endpoint))
    leg = np.stack(leg, axis=-1)
    return leg

# High symmetry points for Si
G =  np.array([0, 0, 0])
L =  np.array([1/2, 1/2, 1/2])
K =  np.array([3/4, 3/4, 0])
X =  np.array([0, 0, 1])
W =  np.array([1, 1/2, 0])
U =  np.array([1/4, 1/4, 1])


number_of_points_k_space = 100
# The path for k vector (x axis)
Lambda = path(L, G, number_of_points_k_space, endpoint=False)
Delta = path(G, X, number_of_points_k_space, endpoint=False)
X_uk = path(X, U, number_of_points_k_space / 4, endpoint=False)
Sigma = path(K, G, number_of_points_k_space, endpoint=True)

k_space = np.vstack((Lambda, Delta, X_uk, Sigma))

# %% Reciprocal space for the lattice

# I decided to redefine funkction for genereting reciprocal vectors G because 
# this implementation is not centered in (0,0,0). It is translated to (3,0,0)
# and the order of the points is different. I do not undrstand how to translate
# to our space as the autor of the git code. It is taken from mathemitica cod.

def G_vectors(states, reciprocal_basis):

    n = states**3
    
    # internal cached implementation
    @functools.lru_cache(maxsize=n)
    def coefficients(m):
        n = (states**3) // 2
        s = m + n
        floor = states // 2

        h = s // states**2 - floor
        k = s % states**2 // states - floor
        l = s % states - floor

        return h, k, l
    
    
    n = states**3
    G = np.zeros(shape=(n, n, 3))
    
    for i in range(n):
        for j in range(n):
            if i == j:                
                G[i,j, :] = coefficients(i - n // 2) @ reciprocal_basis

                
            else:
                G[i,j, :] = coefficients(i - j) @ reciprocal_basis
                        

        
    return G
    
    
    
G = G_vectors(no_of_states, reciprocal_base)    

# %% Defining Hamiltonian matrix


def Hamiltonian(form_factors, G, k, tau, states):
    
    
    n = states**3
    H = np.zeros(shape=(n, n))
    

    for i in range(n):
        for j in range(n):
            if i == j:
                H[i][j] =  SCALE * KINETIC_CONSTANT * ((k[0] + G[i,j, 0])**2 + (k[1] + G[i,j, 1])**2 + (k[2] + G[i,j, 2])**2)
                
            else:
             
                factors = form_factors.get(G[i,j,0]**2 + G[i,j,1]**2+ G[i,j,2]**2) # magnitude of the vector must be 0,3,4,11
                if factors:
                    # only the symmetry part
                    H[i][j] =  factors * np.cos(2 * np.pi * (G[i,j, 0]* tau[0] + G[i,j, 1]* tau[1] + G[i,j, 2]* tau[2])) # + 1j * asym * np.sin(2 * np.pi * g @ tau)
                else:
                    H[i][j] = 0
                        

        
    return H


# Calculating the structure for each k vector for every G points
def calculate_structure(form_factors, G, path, tau, states):
    
    energies = []
    

    for k in path:

        H = Hamiltonian(form_factors, G, k, tau, states)
        eigvals = np.linalg.eigvals(H)
        eigvals.sort()
        energies.append(eigvals[:8]) # take only first 8 bands
        
    energies = np.stack(energies, axis=-1)

    
    return energies

bands = calculate_structure(form_factors, G, k_space, tau, no_of_states)

# %% Plotting

def plot_energies(bands):
    

    # offset the bands so that the top of the valence bands is at zero
    bands -= max(bands[3])

    plt.figure(figsize=(10, 6))


    plt.xlim(0, len(bands))
    # bands[0] - wartosci dla 1 pasma
    # max(bands[7]) - wartosci dla ostatniego pasma
    plt.ylim(min(bands[0]) - 1, max(bands[7]) + 1)

    # custom tick names for k-points
    xticks = number_of_points_k_space* np.array([0, 0.5, 1, 1.5, 2, 2.25, 2.75, 3.25])
    plt.xticks(xticks, ('$L$', '$\Lambda$', '$\Gamma$', '$\Delta$', '$X$', '$U,K$', '$\Sigma$', '$\Gamma$'))


    plt.xlabel('k-Path')
    plt.ylabel('E(k) [eV]')

    plt.title('Krzem')


    for band in bands:
        plt.plot(band, lw=2.0)
    
    plt.show()
    
plot_energies(bands)




