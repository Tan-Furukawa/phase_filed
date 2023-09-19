#%%
import numpy as np

def prepare_fft(Nx, Ny, dx, dy):
    Nx21 = Nx // 2 + 1
    Ny21 = Ny // 2 + 1

    Nx2 = Nx + 2
    Ny2 = Ny + 2

    delkx = (2.0 * np.pi) / (Nx * dx)
    delky = (2.0 * np.pi) / (Ny * dy)

    kx = np.zeros(Nx2)
    ky = np.zeros(Ny2)

    for i in range(1, Nx21 + 1):
        fk1 = (i - 1) * delkx
        kx[i - 1] = fk1
        kx[Nx2 - i - 1] = -fk1

    for j in range(1, Ny21 + 1):
        fk2 = (j - 1) * delky
        ky[j - 1] = fk2
        ky[Ny2 - j - 1] = -fk2

    kx = kx[:Nx]
    ky = ky[:Ny]

    k2 = np.zeros((Nx, Ny))
    k4 = np.zeros((Nx, Ny))

    for i in range(Nx):
        for j in range(Ny):
            k2[i, j] = kx[i] ** 2 + ky[j] ** 2
            k4[i, j] = k2[i, j] ** 2

    return kx, ky, k2, k4

# # 使い方の例
# Nx = 8
# Ny = 8
# dx = 1.0
# dy = 1.0
# kx, ky, k2, k4 = prepare_fft(Nx, Ny, dx, dy)

#%%
