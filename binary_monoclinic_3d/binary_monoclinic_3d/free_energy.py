#%%
# import numpy as np
import cupy as cp


def get_free_energy(con, w_ab, w_or):

    w = con * w_ab + (1 - con) * w_or
    ww = w_ab - w_or

    def get_dfdcon(con):
        dfdcon = w*(1-2*con) + con*(1-con)*ww + \
            (cp.log(con) - cp.log(1-con))
        return dfdcon

    # cp.set_printoptions(precision=15)
    # con = cp.clip(con, 0.00001, 0.9999)
    # A = 1.0
    # dfdcon = A * (2.0 * con * (1 - con)**2 - 2.0 * con**2 * (1.0 - con))

    min_c = 0.001
    max_c = 0.999

    dfdcon = cp.zeros(con.shape)
    dfdcon = get_dfdcon(con)
    dfdcon[con < min_c] = (get_dfdcon(min_c))[con < min_c]
    dfdcon[con > max_c] = (get_dfdcon(max_c))[con > max_c]
    # print(get_dfdcon(con[cp.logical_and(min_c < con,con < max_c)]))

    g = w * con * (1-con) + (con * cp.log(con) + (1-con) * cp.log(1-con))

    return dfdcon, g


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np

    cp.random.seed(seed=0)
    con = cp.random.rand(4, 4)
    R = 8.31
    P = 10 ** 5
    T = 600 + 273
    w_ab = (22820 - 6.3 * T + 0.461 * P / 10 ** 5) / (R * T)
    w_or = (19550 - 10.5 * T + 0.327 * P / 10 ** 5) / (R * T)

    dg, g = get_free_energy(con, w_ab, w_or)
    print(dg)

    print("----------------------")

    con[0,0] = -1.0
    con[0,1] = -1.0
    con[1,0] = 2.0
    con[1,1] = 2.0
    dg, g = get_free_energy(con, w_ab, w_or)
    print(dg)

# %%
