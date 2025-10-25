#Wolny elektron w sześciennej sieci w przestrzeni k
#Jednostki atomowe: \hbar = 1, e = 1, m = 1

#H = 1/2m *k^2 - hamiltonian w przestrzeni k

import numpy as np
from numpy import linalg as LA

#Parametry

Lx = 10  #długość pudełka w x
L = Lx**3  #objętość pudełka to się przyda przy normalizacji
Nk = 6  #liczba punktów k w każdą stronę
 # 


#Budowanie siatki k-punktów z okresowymi warunkami brzegowymi
def build_k_grid(Nk, Lx):
    n_vals = np.arange(-Nk//2, Nk//2) 
    klist = []
    #tworzenie k gdzie k-ity werktor falowy to 2π*ni/Lx 
    for kz in n_vals:
        for ky in n_vals:
            for kx in n_vals:
                kx_val = 2.0 * np.pi * kx / Lx
                ky_val = 2.0 * np.pi * ky / Lx
                kz_val = 2.0 * np.pi * kz / Lx
                klist.append([kx_val, ky_val, kz_val])
    return np.array(klist, dtype=float)
kvecs = build_k_grid(Nk, Lx) # wektory falowe k
print(kvecs)


#Budowanie macierzy Hamiltonianu w bazie fal płaskich
def Macierz(kvecs):
    M = kvecs.shape[0]
    #print(M)
    H = np.zeros((M, M))
    # Obliczanie jaki parametr m wchodzić do macierzy
    k2 = np.sum(kvecs**2, axis=1) #k2 = x^2 + y^2 + z^2
    kin = 1/2 * k2
    #budowanie macierzy Hamiltonianu
    for i in range(M):
        for j in range(M):
            if i == j:
                H[i, j] = kin[i]
    return H
M = Macierz(kvecs)

E = LA.eigvalsh(M) #E - wartosci wlasne ważne abu dawać końcówe sh bo inaczej będą zespolone wartości własne
print(E)