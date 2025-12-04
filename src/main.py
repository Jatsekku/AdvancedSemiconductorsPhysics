# This code calculate dispersion relation for a single electron in a cube 
# with the length Lx. The Hamiltonian is given by H = 1/2m *k^2. 
# Atomic units: \hbar = 1, e = 1, m = 1

import math
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

# Building k space with periodic boundary conditions 
def generate_k_space(nk, Lx):
    n_vals = np.arange(-nk, nk+1) 
    k_vectors = []

    for kz in n_vals:
        for ky in n_vals:
            for kx in n_vals:
                kx_val = 2.0 * np.pi * kx / Lx
                ky_val = 2.0 * np.pi * ky / Lx
                kz_val = 2.0 * np.pi * kz / Lx

                k_vectors.append([kx_val, ky_val, kz_val])

    return np.array(k_vectors, dtype=float)

# Building reciprocal lattice vector G = n * 2 pi/a
def generate_reciprocal_lattice(ngx, ngy, ngz, a):
    G_vectors = []

    for gz in np.arange(-ngz, ngz+1):
        for gy in np.arange(-ngy, ngy+1):
            for gx in np.arange(-ngx, ngx+1):

                G_vectors.append([gx*np.pi/a,gy*np.pi/a,gz*np.pi/a])

    return np.array(G_vectors, dtype=float)

def plot_3d_space(k_space):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    kx = k_space[:, 0]
    ky = k_space[:, 1]
    kz = k_space[:, 2]

    ax.scatter(kx, ky, kz, s=5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

def potential(g, tau, ff):
    return ff * np.cos(2 * np.pi * g @ tau)

# Building the Hamiltonian matrix in the base of plane wave 
def Hamiltonian(kx, ky, kz, reciprocal_lattice, f_factors, tau):
    M = reciprocal_lattice.shape[0]
    H = np.zeros((M, M))

    G = reciprocal_lattice
    Gx, Gy, Gz = G.T
    
    for i in range(M):
            # Diagonal
            H[i,i] = 0.5*((kx+Gx[i])**2 +
                          (ky+Gy[i])**2 +
                          (kz+Gz[i])**2)

            for j in range(M):
                # Rest
                if i != j:
                    delta_G = G[i] - G[j]
                    #print(f'Delta G: {delta_G}')
                    idk = round(np.dot(delta_G, delta_G))
                    #print(f'idk: {idk}')
                    ff = f_factors.get(idk)
                    #print(f'ff: {ff}')
                    x = potential(delta_G, tau, ff) if ff else 0
                    print(x)
                    #print(f'X: {x}')
                    H[i,j] = 0

    return H


# def main():
# Parameters
Lx = 10         # A, the length of the cube
L = Lx**3       # A^3, the volume of the cube (important diuring normalization)
Nk = 2          # the number of the points in k space
a = 5.65325     # A, Should not be equal Lx? 
nG = 2
tau = 1/2 * np.array([1, 1, 1])



form_factors = {3.0: np.array(-0.21)* 13.6059,
                8.0: np.array(0.04)* 13.6059, 
                11.0: np.array(0.08)* 13.6059}

# Generate k space
k_space = generate_k_space(Nk, Lx)
plot_3d_space(k_space)

#Generate reciprocal lattice
G = generate_reciprocal_lattice(nG, nG, nG, a)
plot_3d_space(G)

G_dot_product = []
for i in range(len(G)):
    G_dot_product.append(G[i, :] @ G[1, :])
    


fig, ax = plt.subplots()

kx = np.linspace(-np.pi/a, np.pi/a)

for kxi in kx:
    H = Hamiltonian(kxi,0.0,0.0, G, form_factors, tau)    
    E = LA.eigvalsh(H)                      # only real eigenvalues of energy
    Eigen_val_E = list(dict.fromkeys(E))    # unique values of E
    kxi_vecs = np.ones((1, len(Eigen_val_E)))
    plt.scatter(kxi*kxi_vecs, Eigen_val_E, c='blue', marker = '.')
    

ticks = np.linspace(-np.pi/a, np.pi/a, 6)
ax.set_xticks(ticks)
tick_labels = [fr"{val/np.pi*a:.1f}$\pi/a$" for val in ticks]
ax.set_xticklabels(tick_labels)

# etykiety osi
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$E(k_x)$")
plt.grid()


plt.show()
