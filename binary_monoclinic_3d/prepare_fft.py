#%%
import numpy as np
from _plot import dim3_plot as myplt3

def prepare_fft(Nx, Ny, Nz, dx, dy, dz):
    Nx21 = Nx // 2 + 1
    Ny21 = Ny // 2 + 1
    Nz21 = Nz // 2 + 1

    Nx2 = Nx + 2
    Ny2 = Ny + 2
    Nz2 = Nz + 2

    delkx = (2.0 * np.pi) / (Nx * dx)
    delky = (2.0 * np.pi) / (Ny * dy)
    delkz = (2.0 * np.pi) / (Nz * dz)

    kx = np.zeros(Nx2)
    ky = np.zeros(Ny2)
    kz = np.zeros(Nz2)

    for i in range(1, Nx21 + 1):
        fk1 = (i - 1) * delkx
        kx[i - 1] = fk1
        kx[Nx2 - i - 1] = -fk1

    for j in range(1, Ny21 + 1):
        fk2 = (j - 1) * delky
        ky[j - 1] = fk2
        ky[Ny2 - j - 1] = -fk2

    for k in range(1, Nz21 + 1):
        fk3 = (k - 1) * delkz
        kz[k - 1] = fk3
        kz[Nz2 - k - 1] = -fk3

    kx = kx[:Nx]
    ky = ky[:Ny]
    ky = kz[:Nz]

    k2 = np.zeros((Nx, Ny, Nz))
    k4 = np.zeros((Nx, Ny, Nz))

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                k2[i, j, k] = kx[i] ** 2 + ky[j] ** 2 + kz[k] ** 2
                k4[i, j, k] = k2[i, j, k] ** 2

    return kx, ky, kz, k2, k4

# Nx = 8
# Ny = 8
# Nz = 8
# dx = 1.0
# dy = 1.0
# dz = 1.0
# kx, ky, kz, k2, k4 = prepare_fft(Nx, Ny, Nz, dx, dy, dz)
# myplt3.display_3d_matrix(k2)

#%%
