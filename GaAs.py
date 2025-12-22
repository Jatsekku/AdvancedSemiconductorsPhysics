import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

def wektoryG(Nrange, a, Gcut=None):
    two_pi_over_a = 2.0 * np.pi / a
    # wektory odwrotnej sieci
    b1 = two_pi_over_a * np.array([-1.0,  1.0,  1.0])
    b2 = two_pi_over_a * np.array([ 1.0, -1.0,  1.0])
    b3 = two_pi_over_a * np.array([ 1.0,  1.0, -1.0])

    Gs = []
    seen = []
    tol = 1e-8
    for h in range(-Nrange, Nrange + 1):
        for k in range(-Nrange, Nrange + 1):
            for l in range(-Nrange, Nrange + 1):
                G = h * b1 + k * b2 + l * b3
                gnorm = np.linalg.norm(G)
                if Gcut is not None and gnorm > Gcut:
                    continue
                # usuwanie duplikatów
                key = tuple(np.round(G, 8))
                if any(np.allclose(G, s, atol=tol) for s in seen):
                    continue
                seen.append(G)
                Gs.append(G)
    Gs = np.array(Gs, dtype=float)
    # sortowanie według |G|
    idx = np.argsort(np.sum(Gs**2, axis=1))
    return Gs[idx]

#prosty model form-faktora atomowego

def formfactor():
   #symetrczyna Vs
   form_factors_symetryczne = {
        3: -0.252 * 13.6059,
        8: 0 * 13.6059,
        11: 0.08 * 13.6059
    }
   form_factors_asymetryczne = {
        3: -0.68 * 13.6059,
        8: 0.066 * 13.6059,
        11: 0.012 * 13.6059
    }
   return form_factors_symetryczne,form_factors_asymetryczne

#Hamitlotnian
def Hamiltonian(kvec, gvecs, a, form_factors_symetryczne, form_factors_asymetryczne):
    M = gvecs.shape[0]
    H = np.zeros((M, M), dtype=float)

    # pozycje dwóch atomów w komórce elementarnej
    tau_Ga = np.array([0.0, 0.0, 0.0]) * a
    tau_As = np.array([0.25, 0.25, 0.25]) * a
    taus = [tau_Ga, tau_As]

    #Stała enrgii kinetycznej
    prefactor = 3.80998212  # eV·Å^2

    #część kinetyczna
    for i in range(M):
        Gvec = gvecs[i]
        k_plus_G = kvec + Gvec
        kinetic_energy = prefactor * np.dot(k_plus_G, k_plus_G)
        H[i, i] = kinetic_energy

    #część potencjalna

    norm_factor = (np.pi / a) ** 2
    for i in range(M):
        for j in range(i + 1, M):
            Gdiff = gvecs[j] - gvecs[i]
            n2 = int(round(np.dot(Gdiff, Gdiff) / norm_factor))
            Vg_sym = form_factors_symetryczne.get(n2, 0.0)  # w eV
            Vg_asym = form_factors_asymetryczne.get(n2, 0.0)  # w eV
            if Vg_sym != 0.0 or Vg_asym != 0.0:
                phase = np.dot(Gdiff, tau_As)
                S_real = 1.0 + np.cos(phase)  # część rzeczywista
                S_imag = -np.sin(phase)       # część urojona
                Vij_sym = Vg_sym * (S_real / 2.0)
                Vij_asym = Vg_asym * (S_imag / 2.0)
                Vij = Vij_sym + Vij_asym
            else:
                Vij = 0.0
            H[i, j] = Vij
            H[j, i] = Vij
    return H

#tworzenie ścieżki w strefie Brillouina
def kpath(kpoints_frac, npoints_per_seg, a):
    two_pi_over_a = 2.0 * np.pi / a
    k_cart = [two_pi_over_a * np.array(p) for p in kpoints_frac]
    kpath = []
    kdist = []
    dist = 0.0
    for p in range(len(k_cart) - 1):
        start = k_cart[p]
        end = k_cart[p + 1]
        # avoid duplicate endpoints between consecutive segments
        pts = np.linspace(0, 1, npoints_per_seg, endpoint=(p == len(k_cart) - 2))
        for t in pts:
            k = start * (1 - t) + end * t
            if len(kpath) == 0:
                kpath.append(k)
                kdist.append(0.0)
            else:
                prev = kpath[-1]
                kpath.append(k)
                dist += np.linalg.norm(k - prev)
                kdist.append(dist)
    return np.array(kpath), np.array(kdist)


# wyrównanie do VBM 
def align_to_vbm(energies, valence_bands=4, target_gap=None):
    vbm = np.max(energies[:, :valence_bands])
    shifted = energies - vbm
    scissor = 0.0
    if target_gap is not None and valence_bands < energies.shape[1]:
        cmin = np.min(shifted[:, valence_bands:])
        scissor = max(target_gap - cmin, 0.0)
        if scissor > 0:
            shifted[:, valence_bands:] += scissor
    return shifted, vbm, scissor

def main():
    a = 5.65325  # Å
    # enlarged basis
    Nrange = 7
    Gcut = 4.0 * (2.0 * np.pi / a)
    G = wektoryG(Nrange, a, Gcut=Gcut)
    M = G.shape[0]
    print(f"Using {M} plane waves (Nrange={Nrange}, Gcut={Gcut:.2f})")
    f_sym, f_asym = formfactor()

    # extended high-symmetry path: L–Γ–X–W–K–Γ
    k_frac = [
        (0.5, 0.5, 0.5),   # L
        (0.0, 0.0, 0.0),   # Γ
        (1.0, 0.0, 0.0),   # X
        (1.0, 0.5, 0.0),   # W
        (0.75, 0.75, 0.0), # K
        (0.0, 0.0, 0.0)    # Γ
    ]
    labels = ['L', 'Γ', 'X', 'W', 'K', 'Γ']
    nseg = 80
    kpath_vals, kdist = kpath(k_frac, nseg, a)
    Nk = kpath_vals.shape[0]

    energies = np.zeros((Nk, M))
    for ik, kvec in enumerate(kpath_vals):
        H = Hamiltonian(kvec, G, a, f_sym, f_asym)
        energies[ik, :] = LA.eigvalsh(H)

    # shift to VBM = 0 and apply optional scissor to ~1.42 eV gap
    shifted, vbm, scissor = align_to_vbm(energies, valence_bands=4, target_gap=1.42)
    print(f"VBM shift: {vbm:.3f} eV; scissor applied: {scissor:.3f} eV")

    fig, ax = plt.subplots(figsize=(9, 6))
    for band in range(M):
        ax.plot(kdist, shifted[:, band], color='C0', linewidth=0.8)

    seg_indices = [0]
    for s in range(1, len(k_frac)):
        seg_indices.append(min(s * (nseg - (1 if s == len(k_frac) - 1 else 0)), Nk - 1))
    seg_x = kdist[seg_indices]
    for sx in seg_x:
        ax.axvline(sx, color='k', lw=0.6, alpha=0.6)
    ax.set_xticks(seg_x)
    ax.set_xticklabels(labels)

    ax.set_xlabel("k-path")
    ax.set_ylabel("Energy (eV, VBM=0)")
    ax.set_title("GaAs — pseudopotencjał, L–Γ–X–W–K–Γ")
    ax.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()