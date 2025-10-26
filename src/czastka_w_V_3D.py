# -*- coding: utf-8 -*-
"""
Zadaniem tego programu będzie policzenie energii własnych dla częstki
w przestrzeni jednowymiarowej o długosci L z dodatkowym potencjałem V. 
Przestrzeń zdefiniowana jest w taki sposób, że y = x/L, y = [0,1], tzn. 
poruszamy się w przeskalowanej przestrzeni x/L. Dodatkowo z warunków 
brzegowych wiemy, że funkcja falowa f(y=0) = f(y = 1) = 0 (funkcja 
falowa zanika na ograniczeniu naszej przestrzeni). 

Wygenerowane na podstawie: https://www.youtube.com/watch?v=ay0zZ8SUMSk

@author: Magdalena Wietrzyńska
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import eigsh


N = 50                          # liczba punktów w przestzreni
dy = 1/N                        # odległosć między punktami 
y = np.linspace(0, 1, N+1)      # wektor odległosci 
m =  1                          # masa cząstki
L = 100                         # wielkosc badanego przedziału, powinna być duża


X, Y, Z = np.meshgrid(y, y, y, indexing='ij' )      # 'ij' - numeruj jak w macierzy


V = (X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2      # potencjał harminiczny 3D


# Budujemy macierz
d = 1/dy**2 * np.ones(N-1)                  # elementy na diagonali
e = -1/(2*dy**2) * np.ones(len(d)-1)        # elementy poza diagonalą 
M1D = diags([e,d,e], offsets= [-1, 0, 1])   # macierz z diagonali 
I = eye(N-1)                                # macierz jednostkowa
H_x = kron(kron(M1D, I), I)                 # tego nie do końca rozumiem jak to się miesza
H_y = kron(kron(I, M1D), I)
H_z = kron(kron(I, I), M1D)

H3D = H_x + H_y + H_z
V_flattened = (m*L**2*V[1:-1,1:-1,1:-1]).ravel()
H = H3D + diags(V_flattened)

eigen_energy, eigen_functions = eigsh(H, k=10, which='SM') # liczba wartoci zwracanych, SM - smallest
psi0 = eigen_functions[:,0].reshape((N-1, N-1, N-1))
plt.imshow(psi0[:,:,N//2])
plt.title("Funkcja falowa – przekrój XY")
plt.colorbar()
plt.show()

plt.figure()
psi1 = eigen_functions[:,1].reshape((N-1, N-1, N-1))
plt.imshow(psi1[:,:,N//2])
plt.title("Funkcja falowa – przekrój XY")
plt.colorbar()
plt.show()









d = 1/dy**2 + L**2*m*V[1:-1]            # elementy na diagonali
e = -1/(2*dy**2) * np.ones(len(d)-1)    # elementy poza diagonalą 
# eigen_energy, eigen_functions = eigh_tridiagonal(d, e)



# plt.figure()
# plt.plot(eigen_functions.T[0] + eigen_energy[0]/(m*L**2), label = 'f1')
# plt.plot(eigen_functions.T[1] + eigen_energy[1]/(m*L**2), label = 'f2')
# plt.plot(eigen_functions.T[2] + eigen_energy[2]/(m*L**2), label = 'f3')
# plt.plot(eigen_functions.T[3] + eigen_energy[3]/(m*L**2), label = 'f4')
# plt.plot(eigen_functions.T[4] + eigen_energy[4]/(m*L**2), label = 'f5')
# plt.legend()
