#%%
# import numpy as np
import cupy as cp


def free_energ_ch_v2(con, w_ab, w_or):

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

    # dfdcon = cp.zeros(con.shape)
    dfdcon = get_dfdcon(con)
    dfdcon[con < min_c] = get_dfdcon(min_c)
    dfdcon[con > max_c] = get_dfdcon(max_c)

    # dfdcon = -w * (2*con) + (cp.log(con) - cp.log(1-con))
    g = w * con * (1-con) + (con * cp.log(con) + (1-con) * cp.log(1-con))

    return dfdcon, g


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import numpy as np

    cp.random.seed(seed=0)
    con = cp.random.rand(8, 8)
    P = 10 ** 5
    T = 600 + 273
    w_ab = (22820 - 6.3 * T + 0.461 * P / 10 ** 5) / (R * T)
    w_or = (19550 - 10.5 * T + 0.327 * P / 10 ** 5) / (R * T)


    free_energ_ch_v2(con, w_ab, w_or)


    x = np.linspace(0.001, 0.999, 100)
    w = x * w_ab + (1 - x) * w_or
    g = x*np.log(x) + (1-x)* np.log(1-x) + w*x*(1-x)
    plt.plot(x, g)

# %%
