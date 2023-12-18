#%%
# from matplotlib.colors import Normalize
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from micro_ch_pre import micro_ch_pre
from prepare_fft import prepare_fft
from green_tensor import green_tensor
from solve_elasticity import solve_elasticity
from free_energ_ch_v2 import free_energ_ch_v2
# import matplotlib.pyplot as plt
import time
from _plot import dim3_plot as myplt3
from _save import save_3d_plot as save

if __name__ == "__main__":
    save.create_directory("result")
    dirname = f"{save.make_dir_name()}/res"
    save.create_directory(f"result/{dirname}")

    Nx = 64; Ny = 64; Nz = 64
    dx = 1.0; dy = 1.0; dz = 1.0
    nstep = 10000
    nsave = 10
    nprint = 100
    dtime = 5.0e-2
    coefA = 1.0
    c0 = 0.4
    mobility = 1.0
    grad_coef = 2.0
    noise = 0.3
    R = 8.31446262
    P = 1 * 10**5 # [Pa]
    T = 800 # [K]

    # eigen strains 
    # ei0_ij * c(r)
    #--------------------------------------------------
    # feldspar
    #--------------------------------------------------
    ei0 = cp.array([0.0543, 0.0115, 0.0110, 0, 0.0131, 0])
    # ei0 = np.array([0.0567, 0, 0.016858, 0, 0.016896, 0]) #Robin 1974

    w_or = (22820 - 6.3 * T + 0.461 * P / 10 ** 5) / (R * T)
    w_ab = (19550 - 10.5 * T + 0.327 * P / 10 ** 5) / (R * T)

    N_A = 6.02 * 10.0 ** 23
    v_or = 8.60 * 13.2  * 7.18 * cp.sin(116 / 180 * cp.pi)
    n_or = 1 / (v_or * 10**(-30)) * 4 / N_A
    v_ab = 8.15 * 12.85 * 7.12 * cp.sin(116 / 180 * cp.pi)
    n_ab = 1 / (v_ab * 10**(-30)) * 4 / N_A
    n0 = n_or * c0 + (1 - c0) * n_ab

    # Cij[GPa] * 10^9 * v[Å] * 10*(-30) * NA[/mol] = [/mol]
    cm = cp.array([
        [ 93.9,  41.5,  52.2,    0, -26.2,    0],
        [ 41.5, 176.8,  23.1,    0,  14.2,    0],
        [ 52.2,  23.1,  82.1,    0, -19.5,    0],
        [    0,     0,     0, 17.8,     0,  9.7],
        [-26.2,  14.2, -19.5,    0,  44.2,    0],
        [    0,     0,     0,  9.7,     0, 35.0]
    ]) * 10.0**9 / (R * T) / n0

    # con = cpのmol濃度
    c_p = cp.array([
        [ 93.9,  41.5,  52.2,    0, -26.2,    0],
        [ 41.5, 176.8,  23.1,    0,  14.2,    0],
        [ 52.2,  23.1,  82.1,    0, -19.5,    0],
        [    0,     0,     0, 17.8,     0,  9.7],
        [-26.2,  14.2, -19.5,    0,  44.2,    0],
        [    0,     0,     0,  9.7,     0, 35.0]
    ]) * 10.0**9 / (R * T) / n0

    # applied strains
    ea = cp.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
        ])

    energy_g = cp.zeros(nstep) + cp.nan
    energy_el = cp.zeros(nstep) + cp.nan

    # initialize stress
    s11 = cp.zeros((Nx, Ny, Nz))
    s22 = cp.zeros((Nx, Ny, Nz))
    s12 = cp.zeros((Nx, Ny, Nz))

    # initialize strain
    e11 = cp.zeros((Nx, Ny, Nz))
    e22 = cp.zeros((Nx, Ny, Nz))
    e12 = cp.zeros((Nx, Ny, Nz))

    cp.random.seed(123)
    con = micro_ch_pre(Nx, Ny, Nz, c0, noise)
    # con = np.load('../data/con1.npy')
    bulk = cp.mean(con)

    kx, ky, kz, k2, k4 = prepare_fft(Nx, Ny, Nz, dx, dy, dz)
    # %%
    # tmatx, omeg11 = green_tensor(
    #      Nx,Ny,Nz,
    #      cp.asnumpy(kx),cp.asnumpy(ky),cp.asnumpy(kz),
    #      cp.asnumpy(c_p),cp.asnumpy(cm))
    # np.save("../data/tmatx_3d_2023.npy", tmatx)
    tmatx = cp.asarray(np.load("../data/tmatx_3d_2023.npy"))
    # %%

for istep in range(1, nstep + 1): 
        print(istep)
        
        # Calculate derivatives of free energy and elastic energy
        delsdc, s, el = solve_elasticity(tmatx,cm,c_p,ea,ei0,con,c0)
        # delsdc = 0

        # Assuming you have the free_energ_ch_v2 and solve_elasticity_v2 functions
        dfdcon, g = free_energ_ch_v2(con,w_ab, w_or)

        energy_g[istep-1] = cp.sum(g)
        # energy_el[istep-1] = np.sum(el)

        conk = cp.fft.fftn(con)
        dfdconk = cp.fft.fftn(dfdcon)
        delsdck = cp.fft.fftn(delsdc)
        
        # Time integration
        numer = dtime * mobility * k2 * (dfdconk + delsdck)
        # numer = dtime * mobility * k2 * (dfdconk)
        denom = 1.0 + dtime * coefA * mobility * grad_coef * k4
        conk = (conk - numer) / denom
        con = cp.real(cp.fft.ifftn(conk))

        if (istep % nprint == 0) or (istep == 1) or (np.mean(con)/bulk <0.99):
            con_disp = np.flipud(np.asnumpy(con.transpose()))
            # plt.imshow(con_disp)の図の向きは、
            # y
            # ↑
            # |
            # + --→ x [100]
            # となる。
            myplt3.display_3d_matrix(np.asnumpy(con_disp))
        
        if (istep % nsave == 0) or (istep == 1) or (np.mean(con)/bulk <0.99):
            np.save(f"result/{dirname}/con_{istep}.npy", cp.asnumpy(con))
# %%
