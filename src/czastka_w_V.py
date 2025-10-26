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
from scipy.linalg import eigh_tridiagonal


N = 100                         # liczba punktów w przestzreni
dy = 1/N                        # odległosć między punktami 
y = np.linspace(0, 1, N+1)      # wektor odległosci 

m =  1                          # masa cząstki
L = 100                         # wielkosc badanego przedziału, powinna być duża

V = (y - 0.5)**2

"""
Nasze zagadnienie wygląda w następujący sposób:
    Macierz M=
    
    | 1/dy^2 + mL^2 V1   -1/(2dy^2)          0             0        0 ...|
    | -1/(2dy^2)      1/dy^2 + mL^2 V2  -1/(2dy^2)         0        0 ...|
    | 0                 -1/(2dy^2)  1/dy^2 + mL^2 V3  -1/(2dy^2)    0 ...|
    ...
    | 0                       0...       -1/(2dy^2)  1/dy^2 + mL^2 V(N-1)|
    
    
    M f = mL^2E f
    f - funkcja falowa

"""


d = 1/dy**2 + L**2*m*V[1:-1]            # elementy na diagonali
e = -1/(2*dy**2) * np.ones(len(d)-1)    # elementy poza diagonalą 
eigen_energy, eigen_functions = eigh_tridiagonal(d, e)



plt.figure()
plt.plot(eigen_functions.T[0] + eigen_energy[0]/(m*L**2), label = 'f1')
plt.plot(eigen_functions.T[1] + eigen_energy[1]/(m*L**2), label = 'f2')
plt.plot(eigen_functions.T[2] + eigen_energy[2]/(m*L**2), label = 'f3')
plt.plot(eigen_functions.T[3] + eigen_energy[3]/(m*L**2), label = 'f4')
plt.plot(eigen_functions.T[4] + eigen_energy[4]/(m*L**2), label = 'f5')
plt.legend()
