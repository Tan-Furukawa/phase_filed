#%%
# from matplotlib.colors import Normalize
import numpy as np
import matplotlib.pyplot as plt
from micro_ch_pre import micro_ch_pre
from prepare_fft import prepare_fft
from green_tensor import green_tensor
from solve_elasticity import solve_elasticity
from free_energ_ch_v2 import free_energ_ch_v2
# import matplotlib.pyplot as plt
import time
# from scipy.fftpack import fft2, ifft2
from _plot import dim3_plot as myplt3
from _save import save_3d_plot as save

if __name__ == "__main__":
    # time0 = time.time()
    save.create_directory("result")
    dirname = f"{save.make_dir_name()}/res"
    save.create_directory(f"result/{dirname}")

    Nx = 64; Ny = 64; Nz = 64
    dx = 1.0; dy = 1.0; dz = 1.0
    nstep = 10000
    nprint = 10
    dtime = 5.0e-2
    ttime = 0.0
    coefA = 1.0
    c0 = 0.4
    mobility = 1.0
    grad_coef = 1.0
    noise = 0.01
    R = 8.31446262
    T = 700
    w = 3.0
    # eigen strains 
    # ei0_ij * c(r)
    #--------------------------------------------------
    # feldspar
    #--------------------------------------------------
    ei0 = np.array([0.0567, 0.0162, 0.016858, 0, 0.016896, 0])
    # ei0 = np.array([0.0567, 0, 0.016858, 0, 0.016896, 0])

    v_or = 8.60 * 13.2  * 7.18 * np.sin(116 / 180 * np.pi)
    v_ab = 8.15 * 12.85 * 7.12 * np.sin(116 / 180 * np.pi)

    # Cij[GPa] * 10^9 * v[Å] * 10*(-30) * NA[/mol] = [/mol]
    # nA + nBでや約分するなら、c0な気がする
    # [Pa/J] = [1/L3]
    cm = np.array([
        [ 93.9,  41.5,  52.2,    0, -26.2,    0],
        [ 41.5, 176.8,  23.1,    0,  14.2,    0],
        [ 52.2,  23.1,  82.1,    0, -19.5,    0],
        [    0,     0,     0, 17.8,     0,  9.7],
        [-26.2,  14.2, -19.5,    0,  44.2,    0],
        [    0,     0,     0,  9.7,     0, 35.0]
    ]) * 10**9 / (R * T) * v_ab * 10**(-30) * 6.02 * 10**23 / 4

    # con = cpのmol濃度
    cp = np.array([
        [ 93.9,  41.5,  52.2,    0, -26.2,    0],
        [ 41.5, 176.8,  23.1,    0,  14.2,    0],
        [ 52.2,  23.1,  82.1,    0, -19.5,    0],
        [    0,     0,     0, 17.8,     0,  9.7],
        [-26.2,  14.2, -19.5,    0,  44.2,    0],
        [    0,     0,     0,  9.7,     0, 35.0]
    ]) * 10**9 / (R * T) * v_or * 10**(-30) * 6.02 * 10**23 / 4

    # applied strains
    ea = np.array([
        [0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00],
        [0.00, 0.00, 0.00]
        ])
    
    json_arr = save.kwargs_to_json(
        Nx = Nx, Ny = Ny, Nz = Nz,
        dx = dx, dy = dy, dz = dz,
        nstep = nstep, nprint = nprint,
        dtime = dtime, ttime = ttime,
        coefA = coefA, c0 = c0,
        mobility = mobility, grad_coef = grad_coef,
        noise = noise,
        R = R, T = T, w = w,
        ei0 = ei0, v_or = v_or, v_ab = v_ab,
        cm = cm, cp = cp, ea = ea
    )

    save.save_json(f"result/{dirname}/_param.json", json_arr)

    energy_g = np.zeros(nstep) + np.nan
    energy_el = np.zeros(nstep) + np.nan

    # initialize stress
    s11 = np.zeros((Nx, Ny, Nz))
    s22 = np.zeros((Nx, Ny, Nz))
    s12 = np.zeros((Nx, Ny, Nz))

    # initialize strain
    e11 = np.zeros((Nx, Ny, Nz))
    e22 = np.zeros((Nx, Ny, Nz))
    e12 = np.zeros((Nx, Ny, Nz))
    # %%
    np.random.seed(123)
    con = micro_ch_pre(Nx, Ny, Nz, c0, noise)
    # con = np.load('../data/con1.npy')
    # %%
    bulk = np.mean(con)

    kx, ky, kz, k2, k4 = prepare_fft(Nx, Ny, Nz, dx, dy, dz)
    # tmatx, omeg11 = green_tensor(Nx,Ny,Nz,kx,ky,kz,cp,cm)
    # np.save("../data/tmatx_3d.npy", tmatx)
    tmatx = np.load("../data/tmatx_3d.npy")

    for istep in range(1, nstep + 1):
        print(istep)
    #     ttime += dtime
        
        # Calculate derivatives of free energy and elastic energy
        # delsdc, s, el = solve_elasticity(Nx,Ny,Nz,tmatx,cm,cp,ea,ei0,con,c0)
        # delsdc = 0

        # Assuming you have the free_energ_ch_v2 and solve_elasticity_v2 functions
        dfdcon, g = free_energ_ch_v2(Nx,Ny,Nz,con,w)

        energy_g[istep-1] = np.sum(g)
        # energy_el[istep-1] = np.sum(el)

        conk = np.fft.fftn(con)
        dfdconk = np.fft.fftn(dfdcon)
        # delsdck = np.fft.fftn(delsdc)
        
        # Time integration
        # numer = dtime * mobility * k2 * (dfdconk + delsdck)
        numer = dtime * mobility * k2 * (dfdconk)
        denom = 1.0 + dtime * coefA * mobility * grad_coef * k4
        conk = (conk - numer) / denom
        con = np.real(np.fft.ifftn(conk))
        
        if (istep % nprint == 0) or (istep == 1) or (np.mean(con)/bulk <0.99):
            np.save(f"result/{dirname}/con_{istep}.npy", con)
            # np.save(f"result/{dirname}/s_{istep}.npy", s)
            # np.save(f"result/{dirname}/el_{istep}.npy", el)
            # plt.imshow(con_disp)の図の向きは、
            # y
            # ↑
            # |
            # + --→ x [100]
            # となる。
            # con_disp = np.flipud(con.transpose())
            # myplt3.display_3d_matrix(con)
    #         # myplt.display_gradient(np.flipud(el.transpose()), isnorm=False)

    plt.colorbar()


    # #%%
    # myplt3.display_3d_matrix(s[:,:,:,1])

    # if __name__ == "__main__":
    #     main()