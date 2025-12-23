# This code calculate dispersion relation for a single electron in a cube 
# with the length Lx. The Hamiltonian is given by H = 1/2m *k^2. 
# Atomic units: \hbar = 1, e = 1, m = 1

import math
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


import functools
import itertools

from scipy import constants as c

# remove constant from function definition so it is not recalculated every time
KINETIC_CONSTANT = c.hbar**2 / (2 * c.m_e * c.e)


# %% Stałe materiałowe
A = 5.43e-10 # Stała sieci dla krzemu [m]
form_factors = {3.0: np.array(-0.21*13.6059), 
               8.0: np.array(0.04*13.6059), 
               11.0:np.array(0.08*13.6059)} 


tau = 1/8 * np.array([1, 1, 1])

# Base vectors in the base of 2 pi / a
b1 = np.array([-1., 1., 1.])
b2 = np.array([1., -1., 1.])
b3 = np.array([1., 1., -1.])   
reciprocal_base= np.array([b1,b2,b3])





# %% Zdefiniowanie drogi dla wektora falowego
def path(start, end, n, endpoint):
    """
    Zwraca w każdym z wymiarów zestaw równooddalonych n ponktów w każdym z wymiarów
    """
    spacings = []
    for i in range(3):
        spacings.append(np.linspace(start[i], end[i], num=int(n), endpoint=endpoint))
    return np.stack(spacings, axis=-1)

#Zdefiniowanie punktów wysokiej symetrii
G =  np.array([0, 0, 0])
L =  np.array([1/2, 1/2, 1/2])
K =  np.array([3/4, 3/4, 0])
X =  np.array([0, 0, 1])
W =  np.array([1, 1/2, 0])
U =  np.array([1/4, 1/4, 1])


number_of_points_k_space = 100
# Scieżka dla wektora falowego
Lambda = path(L, G, number_of_points_k_space, endpoint=False)
Delta = path(G, X, number_of_points_k_space, endpoint=False)
X_uk = path(X, U, number_of_points_k_space / 4, endpoint=False)
Sigma = path(K, G, number_of_points_k_space, endpoint=True)

k_space = np.vstack((Lambda, Delta, X_uk, Sigma))



# %%


def Hamiltonian(lattice_constant, form_factors, reciprocal_basis, k, states, tau):
    
    
    n = states**3
    
    # to jest główny winowajca zamieszania, bo tu jest inaczej trochę niż u nas
    # zdefiniowanie osobno wektora g
    
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
    
    # initializing Hamiltonian matrix
    H = np.zeros(shape=(n, n))
    

    for i, j in itertools.product(range(n), repeat=2):
        if i == j:
            g = coefficients(i - n // 2) @ reciprocal_basis # mnozenie wektorowe?
            v = k + g
            H[i][j] = (2 * np.pi / lattice_constant)**2 * KINETIC_CONSTANT * v @ v
            
        else:
            g = coefficients(i - j) @ reciprocal_basis
            factors = form_factors.get(g @ g) # długosc wektora g jest równa 3,4,11
            # potential is 0 for g**2 != (3, 8, 11)
            if factors:
                
                H[i][j] =  factors * np.cos(2 * np.pi * g @ tau) # + 1j * asym * np.sin(2 * np.pi * g @ tau)
            else:
                H[i][j] = 0
                    

        
    return H



def calculate_structure(lattice_constant, form_factors, reciprocal_basis, states, path, tau):
    
    structure = []
    

    for k in path:
        h = Hamiltonian(lattice_constant, form_factors, reciprocal_basis, k, states, tau)
        eigvals = np.linalg.eigvals(h)
        eigvals.sort()
        # picks out the lowest eight eigenvalues
        print(eigvals[:8])
        structure.append(eigvals[:8]) # bierze pierwsze 8 pasm
    
    
    return np.stack(structure, axis=-1)

bands = calculate_structure(A, form_factors, reciprocal_base, 7, k_space, tau)



# offset the bands so that the top of the valence bands is at zero
bands -= max(bands[3])

plt.figure(figsize=(15, 9))

ax = plt.subplot(111)

# remove plot borders
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# limit plot area to data
plt.xlim(0, len(bands))
plt.ylim(min(bands[0]) - 1, max(bands[7]) + 1)

# custom tick names for k-points
xticks = number_of_points_k_space* np.array([0, 0.5, 1, 1.5, 2, 2.25, 2.75, 3.25])
plt.xticks(xticks, ('$L$', '$\Lambda$', '$\Gamma$', '$\Delta$', '$X$', '$U,K$', '$\Sigma$', '$\Gamma$'), fontsize=18)
plt.yticks(fontsize=18)

# horizontal guide lines every 2.5 eV
for y in np.arange(-25, 25, 2.5):
    plt.axhline(y, ls='--', lw=0.3, color='black', alpha=0.3)

# hide ticks, unnecessary with gridlines
plt.tick_params(axis='both', which='both',
                top='off', bottom='off', left='off', right='off',
                labelbottom='on', labelleft='on', pad=5)

plt.xlabel('k-Path', fontsize=20)
plt.ylabel('E(k) (eV)', fontsize=20)

plt.text(135, -18, 'Fig. 1. Band structure of Si.', fontsize=12)

# tableau 10 in fractional (r, g, b)
colors = 1 / 255 * np.array([
    [31, 119, 180],
    [255, 127, 14],
    [44, 160, 44],
    [214, 39, 40],
    [148, 103, 189],
    [140, 86, 75],
    [227, 119, 194],
    [127, 127, 127],
    [188, 189, 34],
    [23, 190, 207]
])

for band, color in zip(bands, colors):
    plt.plot(band, lw=2.0, color=color)

plt.show()




