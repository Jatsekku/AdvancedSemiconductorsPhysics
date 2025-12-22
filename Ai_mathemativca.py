import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# =============================================================================
# PARAMETERS - Adjust these values
# =============================================================================

# Matrix size parameter (lim=5 gives 124x124 matrix with good accuracy)
lim = 5

# Material parameters
a = 5.6533  # lattice constant in Angstroms
m0 = 9.11e-31  # electron mass in kg
hbarJs = 1.054e-34  # hbar in J-s
hbareV = 6.581e-16  # hbar in eV-s

# Pseudopotentials in Rydberg (multiply by 13.6059 to get eV)
# Adjust these values to fit experimental band gaps
V3s = -0.23  # Symmetric potential for |K|^2 = 3
V4s = 0.0    # Symmetric potential for |K|^2 = 4
V8s = 0.01   # Symmetric potential for |K|^2 = 8
V11s = 0.06  # Symmetric potential for |K|^2 = 11

V3a = 0.07   # Antisymmetric potential for |K|^2 = 3
V4a = 0.05   # Antisymmetric potential for |K|^2 = 4
V8a = 0.0    # Antisymmetric potential for |K|^2 = 8
V11a = 0.01  # Antisymmetric potential for |K|^2 = 11

# =============================================================================
# SETUP
# =============================================================================

# Calculate matrix size
n = lim**3 - 1

# Define basis reciprocal lattice vectors (for FCC)
vecb1 = np.array([-1, 1, 1])
vecb2 = np.array([1, -1, 1])
vecb3 = np.array([1, 1, -1])

# Offset for 2 atoms per unit cell (diamond/zincblende structure)
vecT = np.array([1/8, 1/8, 1/8])

print(f"Matrix size: {n+1}x{n+1}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def h(m):
    """Calculate h index from linear index m"""
    return int(np.floor((m + n/2) / lim**2) - np.floor(lim/2))

def k(m):
    """Calculate k index from linear index m"""
    return int(np.floor(((m + n/2) % lim**2) / lim) - np.floor(lim/2))

def l(m):
    """Calculate l index from linear index m"""
    return int((m + n/2) % lim - np.floor(lim/2))

def K(m):
    """Calculate reciprocal lattice vector for index m"""
    return h(m) * vecb1 + k(m) * vecb2 + l(m) * vecb3

def Vs(m):
    """Symmetric pseudopotential"""
    K_vec = K(m)
    K_squared = np.dot(K_vec, K_vec)
    
    if K_squared == 3:
        return V3s * 13.6059
    elif K_squared == 4:
        return V4s * 13.6059
    elif K_squared == 8:
        return V8s * 13.6059
    elif K_squared == 11:
        return V11s * 13.6059
    else:
        return 0.0

def Va(m):
    """Antisymmetric pseudopotential"""
    K_vec = K(m)
    K_squared = np.dot(K_vec, K_vec)
    
    if K_squared == 3:
        return V3a * 13.6059
    elif K_squared == 4:
        return V4a * 13.6059
    elif K_squared == 8:
        return V8a * 13.6059
    elif K_squared == 11:
        return V11a * 13.6059
    else:
        return 0.0

def V(m):
    """Total pseudopotential (complex)"""
    K_vec = K(m)
    phase = 2 * np.pi * np.dot(K_vec, vecT)
    return Vs(m) * np.cos(phase) + 1j * Va(m) * np.sin(phase)

def A(i, j, veck):
    """Matrix element A_ij for wavevector veck"""
    # Kinetic energy term (diagonal)
    if i == j:
        k_plus_K = veck + K(i)
        kinetic = (hbarJs * hbareV / (2 * m0)) * np.dot(k_plus_K, k_plus_K) * \
                  (2 * np.pi / (a * 1e-10))**2
    else:
        kinetic = 0.0
    
    # Potential energy term
    potential = V(i - j)
    
    return kinetic + potential

# =============================================================================
# K-POINT PATH GENERATOR
# =============================================================================

def kgenerator(m):
    """Generate k-points along high-symmetry path:  L -> Gamma -> X -> W -> K -> Gamma"""
    if m <= 10:  # L (0. 5,0.5,0.5) to Gamma (0,0,0)
        return np.array([0.5 - m/20, 0.5 - m/20, 0.5 - m/20])
    elif 10 < m <= 20:  # Gamma (0,0,0) to X (1,0,0)
        return np.array([(m - 10)/10, 0, 0])
    elif 20 < m <= 25:  # X (1,0,0) to W (1,0. 5,0)
        return np.array([1, (m - 20)/10, 0])
    elif 25 < m <= 30:  # W (1,0.5,0) to K (0.75,0.75,0)
        return np.array([1 - (m - 25)/20, 0.5 + (m - 25)/20, 0])
    else:  # K (0.75,0.75,0) to Gamma (0,0,0)
        return np.array([0.75 - (m - 30)/(40/3), 0.75 - (m - 30)/(40/3), 0])

# =============================================================================
# CALCULATE BAND STRUCTURE AT SYMMETRY POINTS
# =============================================================================

def calculate_bands_at_kpoint(kpoint, num_bands=10):
    """Calculate eigenvalues at a single k-point"""
    # Build Hamiltonian matrix
    matrix_indices = np.arange(-n//2, n//2 + 1)
    matrix_size = len(matrix_indices)
    matrixA = np.zeros((matrix_size, matrix_size), dtype=complex)
    
    for idx_i, i in enumerate(matrix_indices):
        for idx_j, j in enumerate(matrix_indices):
            matrixA[idx_i, idx_j] = A(i, j, kpoint)
    
    # Calculate eigenvalues
    eigenvalues = eigh(matrixA, eigvals_only=True)
    
    # Return top num_bands eigenvalues
    return np.sort(np.real(eigenvalues))[-num_bands:]

# Calculate at high-symmetry points
print("\nCalculating eigenvalues at high-symmetry points.. .\n")

gamma_point = np.array([0.0, 0.0, 0.0])
x_point = np.array([1.0, 0.0, 0.0])
l_point = np.array([0.5, 0.5, 0.5])

print("Gamma point (0,0,0):")
gamma_list = calculate_bands_at_kpoint(gamma_point, 10)
print(gamma_list)

print("\nX point (1,0,0):")
x_list = calculate_bands_at_kpoint(x_point, 10)
print(x_list)

print("\nL point (0.5,0.5,0.5):")
l_list = calculate_bands_at_kpoint(l_point, 10)
print(l_list)

# =============================================================================
# CALCULATE FULL BAND DIAGRAM
# =============================================================================

print("\nCalculating full band diagram...")
print("This may take a few minutes...")

# K-point path:  41 points from m=0 to m=40
k_points = range(41)
num_bands_to_plot = 20
output_bands = []

for m in k_points:
    kpoint = kgenerator(m)
    bands = calculate_bands_at_kpoint(kpoint, num_bands_to_plot)
    
    # Shift energy so highest valence band is at zero (adjust offset as needed)
    bands_shifted = bands - 8.52357
    
    output_bands.append(bands_shifted)
    
    if m % 5 == 0:
        print(f"Progress: {m}/40 k-points completed")

output_bands = np.array(output_bands)

print("\nCalculation complete!")

# =============================================================================
# SAVE TO FILE
# =============================================================================

# Save as CSV (similar to Mathematica export)
df = pd.DataFrame(output_bands)
df.to_csv('gaasband.csv', index=False, header=False)
print("\nBand structure saved to 'gaasband. csv'")

# =============================================================================
# PLOT BAND DIAGRAM
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Plot each band
for band_idx in range(num_bands_to_plot):
    ax.plot(range(41), output_bands[:, band_idx], 'b-', linewidth=1.5)

# Add vertical lines at high-symmetry points
ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax.axvline(x=10, color='k', linestyle='--', alpha=0.3)
ax.axvline(x=20, color='k', linestyle='--', alpha=0.3)
ax.axvline(x=25, color='k', linestyle='--', alpha=0.3)
ax.axvline(x=30, color='k', linestyle='--', alpha=0.3)
ax.axvline(x=40, color='k', linestyle='--', alpha=0.3)

# Add horizontal line at E=0
ax.axhline(y=0, color='r', linestyle='-', alpha=0.5, linewidth=0.5)

# Labels
ax.set_xlabel('k-point', fontsize=12)
ax.set_ylabel('Energy (eV)', fontsize=12)
ax.set_title('Band Structure', fontsize=14)

# Set x-axis labels for high-symmetry points
ax.set_xticks([0, 10, 20, 25, 30, 40])
ax.set_xticklabels(['L', 'Γ', 'X', 'W', 'K', 'Γ'])

ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('band_structure.png', dpi=300)
print("Band diagram saved to 'band_structure.png'")
plt.show()

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*60)
print("BAND STRUCTURE CALCULATION SUMMARY")
print("="*60)
print(f"Material: GaAs (example)")
print(f"Lattice constant: {a} Å")
print(f"Matrix size: {n+1}x{n+1}")
print(f"Number of bands calculated: {num_bands_to_plot}")
print(f"Number of k-points: 41")
print("\nEnergy reference:  Highest valence band at 0 eV")
print("\nFiles generated:")
print("  - gaasband.csv (numerical data)")
print("  - band_structure.png (plot)")
print("="*60)