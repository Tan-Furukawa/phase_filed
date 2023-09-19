import numpy as np

def free_energ_ch_v2(Nx, Ny, con):
    np.set_printoptions(precision=15)

    A = 1.0

    dfdcon = A * (2.0 * con * (1 - con)**2 - 2.0 * con**2 * (1.0 - con))

    return dfdcon