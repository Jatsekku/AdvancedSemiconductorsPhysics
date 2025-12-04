import numpy as np
from numpy.linalg import eigvalsh
import matplotlib.pyplot as plt

# ---------------------------
# Parameters
# ---------------------------
Lx = 10.0           # Cubic lattice length (Å)
a = 5.65325         # Lattice constant (Å)
Nk = 3              # k-points along each direction
nG = 2              # Number of G-vectors in each direction
tau = 0.5 * np.array([1, 1, 1])  # Atomic position in unit cell

# Simple form factors for hydrogen-based lattice (in eV)
form_factors = {
    3: -0.21*13.6059,
    8: 0.04*13.6059,
    11: 0.08*13.6059
}

# ---------------------------
# Generate reciprocal lattice
# ---------------------------
def generate_reciprocal_lattice(nG, a):
    G_vectors = []
    for gz in range(-nG, nG+1):
        for gy in range(-nG, nG+1):
            for gx in range(-nG, nG+1):
                G_vectors.append([gx*2*np.pi/a, gy*2*np.pi/a, gz*2*np.pi/a])
    return np.array(G_vectors, dtype=float)

# ---------------------------
# Potential between plane waves
# ---------------------------
def potential(g, tau, ff):
    return ff * np.cos(2*np.pi * np.dot(g, tau))

# ---------------------------
# Build Hamiltonian matrix
# ---------------------------
def Hamiltonian(kx, ky, kz, G, f_factors, tau):
    M = G.shape[0]
    H = np.zeros((M, M), dtype=float)  # Real Hamiltonian

    for i in range(M):
        H[i, i] = 0.5 * np.sum((np.array([kx, ky, kz]) + G[i])**2)
        for j in range(M):
            if i != j:
                delta_G = G[i] - G[j]
                idk = round(np.dot(delta_G, delta_G))
                ff = f_factors.get(idk)
                H[i, j] = potential(delta_G, tau, ff) if ff is not None else 0
    return H

# ---------------------------
# Generate k-path and compute bands
# ---------------------------
G = generate_reciprocal_lattice(nG, a)

# Simple path along kx from -pi/a to pi/a
kx_path = np.linspace(-np.pi/a, np.pi/a, 10)
bands = []

for kx in kx_path:
    H = Hamiltonian(kx, 0, 0, G, form_factors, tau)
    E = eigvalsh(H)  # Eigenvalues (energies)
    bands.append(E)

bands = np.array(bands)

# ---------------------------
# Plot bands
# ---------------------------
plt.figure(figsize=(6, 4))
for n in range(bands.shape[1]):
    plt.plot(kx_path, bands[:, n], 'b.-', markersize=3)

plt.xlabel(r'$k_x$ (1/Å)')
plt.ylabel('Energy (eV)')
plt.title('Band Structure: Hydrogen-like Cubic Lattice')
plt.grid(True)
plt.show()
