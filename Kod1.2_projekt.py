# This code calculate dispersion relation for a single electron in a cube 
# with the length Lx. The Hamiltonian is given by H = 1/2m *k^2. 
# Atomic units: \hbar = 1, e = 1, m = 1

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

# # Parameters
# Lx = 10         # the length of the cube
# L = Lx**3       # the volume of the cube (important diuring normalization)
# Nk = 6          # the number of the points in k space


# Building k space with periodic boundary conditions 
def build_k_grid(Nk, Lx):
    n_vals = np.arange(-Nk//2, Nk//2) 
    klist = []

    for kz in n_vals:
        for ky in n_vals:
            for kx in n_vals:
                kx_val = 2.0 * np.pi * kx / Lx
                ky_val = 2.0 * np.pi * ky / Lx
                kz_val = 2.0 * np.pi * kz / Lx
                klist.append([kx_val, ky_val, kz_val])
    return np.array(klist, dtype=float)

# Building reciprocal lattice vector G = n * 2 pi/a
def generateGs(ngx,ngy,ngz,a):
    Glist = []
    for nz in np.arange(-ngx//2, ngx//2):
        for ny in np.arange(-ngy//2, ngy//2):
            for nx in np.arange(-ngz//2, ngz//2):
                #G = np.array([nx*np.pi/a,ny*np.pi/a,nz*np.pi/a])
                Glist.append([nx*np.pi/a,ny*np.pi/a,nz*np.pi/a])
    return np.array(Glist)



# Building the Hamiltonian matrix in the base of plane wave 
def Hamiltonian(kx, ky, kz, gvecs, f_factors):
    M = gvecs.shape[0]
    
    H = np.zeros((M, M))
 
    
    for i in range(M):
        for j in range(M):
            if(i==j): 
                H[i,j] = 0.5*((kx-gvecs[i,0])**2 +
                              (ky-gvecs[i,1])**2 +
                              (kz-gvecs[i,2])**2)
            else:
                # anonymous function for potential
                # V = lambda Vs, g, tau: Vs* np.cos(2*np.pi*g @ tau)
                # to uze it write -> V(Vs, g, tau)
                # factors = f_factors.get(gvecs[i, :] @ gvecs[i, :])
                # if factors:
                #     print('huraaa')
                #     H[i,j] = 0
                # else:
                #     H[i,j] = 0.0
                
                H[i,j] = 0.0
    return H


def main():
    # Parameters
    Lx = 10         # A, the length of the cube
    L = Lx**3       # A^3, the volume of the cube (important diuring normalization)
    Nk = 6          # the number of the points in k space
    

    form_factors = {3.0: np.array(-0.21)* 13.6059,
                    8.0: np.array(0.04)* 13.6059, 
                    11.0: np.array(0.08)* 13.6059}
    
    a = 5.65325     # A, Should not be equal Lx? 
    
    kvecs = build_k_grid(Nk, Lx) # k - space
    
    
    # Generate G reciprocal lattice vectors (all of them! ;) 
    G = generateGs(10,10,10, a)
    
    G_dot_product = []
    for i in range(len(G)):
        G_dot_product.append(G[i, :] @ G[1, :])
        
    kx = np.linspace(-np.pi/a, np.pi/a)
    
    
    fig, ax = plt.subplots()
    # plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams['font.size'] = 12
    # plt.tick_params(direction='in')

    for kxi in kx:
        H = Hamiltonian(kxi,0.0,0.0, G, form_factors)    
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
    
    
    plt.show()
    return G_dot_product

    
G_dot_product = main()
