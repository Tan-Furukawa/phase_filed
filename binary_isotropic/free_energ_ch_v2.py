import numpy as np

def free_energ_ch_v2(Nx, Ny, con, w):
    np.set_printoptions(precision=15)

    con = np.clip(con, 0.00001, 0.9999)
    # A = 1.0
    # dfdcon = A * (2.0 * con * (1 - con)**2 - 2.0 * con**2 * (1.0 - con))

    dfdcon = w * (1-2*con) + (np.log(con) - np.log(1-con))
    g = w * con * (1-con) + (con * np.log(con) + (1-con) * np.log(1-con))

    return dfdcon, g
