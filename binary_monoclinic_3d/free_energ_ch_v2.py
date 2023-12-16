import numpy as np


def free_energ_ch_v2(Nx, Ny, Nz, con, w):

    def get_dfdcon(con):
        dfdcon = w * (1-2*con) + (np.log(con) - np.log(1-con))
        return dfdcon

    # np.set_printoptions(precision=15)
    # con = np.clip(con, 0.00001, 0.9999)
    # A = 1.0
    # dfdcon = A * (2.0 * con * (1 - con)**2 - 2.0 * con**2 * (1.0 - con))

    min_c = 0.001
    max_c = 0.999

    dfdcon = np.zeros(con.shape)
    dfdcon[con < min_c] = get_dfdcon(min_c)
    dfdcon[con > max_c] = get_dfdcon(max_c)
    dfdcon[np.logical_and(min_c < con,con < max_c)] = get_dfdcon(con[np.logical_and(min_c < con,con < max_c)])

    # dfdcon = -w * (2*con) + (np.log(con) - np.log(1-con))
    g = w * con * (1-con) + (con * np.log(con) + (1-con) * np.log(1-con))

    return dfdcon, g

#%%

