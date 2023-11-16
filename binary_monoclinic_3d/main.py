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

#%%
time0 = time.time()

Nx = 64
Ny = 64
Nz = 64
NxNy = Nx * Ny * Nz
dx = 1.0
dy = 1.0
dz = 1.0
nstep = 20000
nprint = 1
dtime = 5.0e-2
ttime = 0.0
coefA = 1.0
c0 = 0.4
mobility = 1.0
grad_coef = 1.0
noise = 0.01

energy_g = np.zeros(nstep) + np.nan
energy_el = np.zeros(nstep) + np.nan

# coefA * c * (1-c) + grad_coef * (∇c)^2

# matrix
# cm11 = 1400.0
# cm12 = 600.0
# cm44 = 400.0

R = 8.31446262
T = 700

w = 3
#%%
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
# [Pa/J] = [1/L3]
cm = np.array([
    # [ 93.9,     0,  52.2,    0, -26.2,    0],
    # [    0,     0,     0,    0,     0,    0],
    # [ 52.2,     0,  82.1,    0, -19.5,    0],
    # [    0,     0,     0,    0,     0,    0],
    # [-26.2,     0, -19.5,    0,  44.2,    0],
    # [    0,     0,     0,    0,     0,    0]
    [ 93.9,  41.5,  52.2,    0, -26.2,    0],
    [ 41.5, 176.8,  23.1,    0,  14.2,    0],
    [ 52.2,  23.1,  82.1,    0, -19.5,    0],
    [    0,     0,     0, 17.8,     0,  9.7],
    [-26.2,  14.2, -19.5,    0,  44.2,    0],
    [    0,     0,     0,  9.7,     0, 35.0]
]) * 10**9 / (R * T) * v_ab * 10**(-30) * 6.02 * 10**23 / 4

# con = cpのmol濃度
cp = np.array([
    # [ 93.9,     0,  52.2,    0, -26.2,    0],
    # [    0,     0,     0,    0,     0,    0],
    # [ 52.2,     0,  82.1,    0, -19.5,    0],
    # [    0,     0,     0,    0,     0,    0],
    # [-26.2,     0, -19.5,    0,  44.2,    0],
    # [    0,     0,     0,    0,     0,    0]
    [ 93.9,  41.5,  52.2,    0, -26.2,    0],
    [ 41.5, 176.8,  23.1,    0,  14.2,    0],
    [ 52.2,  23.1,  82.1,    0, -19.5,    0],
    [    0,     0,     0, 17.8,     0,  9.7],
    [-26.2,  14.2, -19.5,    0,  44.2,    0],
    [    0,     0,     0,  9.7,     0, 35.0]
]) * 10**9 / (R * T) * v_or * 10**(-30) * 6.02 * 10**23 / 4
#%%

#--------------------------------------------------
# cubic
#--------------------------------------------------
# a = (0.286 * 10**(-9)) ** 3 * 6.02 * 10**23 / 2

# cp11 = 4.63 * 10**11 / (R * T) * (a)
# cp12 = 1.61 * 10**11 / (R * T) * (a)
# cp44 = 1.09 * 10**11 / (R * T) * (a)
# cp = np.array([
#     [cp11, cp12,    0],
#     [0,    cp11,    0],
#     [0,      0,  cp44]
# ])
# cm11 = 2.33 * 10**11 / (R * T) * (a)
# cm12 = 1.35 * 10**11 / (R * T) * (a)
# cm44 = 1.18 * 10**11 / (R * T) * (a)
# cm = np.array([
#     [cm11, cm12,    0],
#     [0,    cm11,    0],
#     [0,      0,  cm44]
# ])
# ei0 = np.array([
#     [0.05, 0],
#     [0,    0.05],
# ])
#--------------------------------------------------
#%%

# applied strains
ea = np.array([
    [0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00]
    ])

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

# %%
kx, ky, kz, k2, k4 = prepare_fft(Nx, Ny, Nz, dx, dy, dz)
# tmatx, omeg11 = green_tensor(Nx,Ny,Nz,kx,ky,kz,cp,cm)
# myplt3.display_3d_matrix(omeg11)
# plt.imshow(omeg11[:,1,:])
# np.save("../data/tmatx_3d.npy", tmatx)
# %%
tmatx = np.load("../data/tmatx_3d.npy")
# myplt3.display_3d_matrix(tmatx[:,:,:,0,0,0,0])
# plt.imshow(tmatx[:,:,:,0,0,0,0])
# %%
# # %%

for istep in range(1, nstep + 1):
    print(istep)
#     ttime += dtime
    
#     # Calculate derivatives of free energy and elastic energy
    delsdc, el, s = solve_elasticity(Nx,Ny,Nz,tmatx,cm,cp,ea,ei0,con,c0)
    # delsdc = 0

#     # print(delsdc)
    # raise TypeError()

#     # Assuming you have the free_energ_ch_v2 and solve_elasticity_v2 functions
    dfdcon, g = free_energ_ch_v2(Nx,Ny,Nz,con,w)

#     energy_g[istep-1] = np.sum(g)
#     energy_el[istep-1] = np.sum(el)

    conk = np.fft.fftn(con)
    # conk = fftn(con)
    dfdconk = np.fft.fftn(dfdcon)
    delsdck = np.fft.fftn(delsdc)
    
#     # Time integration
    numer = dtime * mobility * k2 * (dfdconk + delsdck)
    denom = 1.0 + dtime * coefA * mobility * grad_coef * k4
    conk = (conk - numer) / denom
    con = np.real(np.fft.ifftn(conk))
    
#     # Clip small deviations
#     # con = np.clip(con, 0.00001, 0.9999)
#     # print(np.mean(con))
#     con = con * (1 + np.sum(con[con < 0])/bulk)
#     con = con * (1 + np.sum(con[con > 1])/bulk)
#     con[con < 0] = 0
#     con[con > 1] = 1
#     # print("----------")
#     # print(bulk)
#     # print(np.sum(con)/ len(con))
#     # # バルク規格化
#     # print("----------")
#     # con = bulk - np.sum(con) / (Nx * Ny) + con
#     # con[0,0] = 0
#     # con[Nx-1,Nx-1] = 1
    
#     # Print results
#     # if ((np.mean(con)/bulk <0.99)):
#     #     raise TypeError()

    if (istep % nprint == 0) or (istep == 1) or (np.mean(con)/bulk <0.99):
        # plt.imshow(con_disp)の図の向きは、
        # y
        # ↑
        # |
        # + --→ x [100]
        # となる。
        # con_disp = np.flipud(con.transpose())
        myplt3.display_3d_matrix(con)
#         # myplt.display_gradient(np.flipud(el.transpose()), isnorm=False)

#         myplt.get_matrix_image(con)

#         plt.imshow(et11)
#         plt.colorbar()
#         plt.show()

#         # np.sum(et11 + ei11)

#         # plt.plot(el[:,1])


# # Calculate compute time
# compute_time = time.process_time() - time0
# print(f"Compute Time: {compute_time:.10f}")

# #%%
myplt3.display_3d_matrix(s[:,:,:,1])