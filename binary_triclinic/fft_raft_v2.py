#%%
from matplotlib.colors import Normalize
import numpy as np
# import matplotlib.pyplot as plt
from micro_ch_pre import micro_ch_pre
from prepare_fft import prepare_fft
from green_tensor import green_tensor
from solve_elasticity_v2 import solve_elasticity_v2
from free_energ_ch_v2 import free_energ_ch_v2
import matplotlib.pyplot as plt
import time
from scipy.fftpack import fft2, ifft2
from _plot import dim2_plot as myplt

#%%
time0 = time.time()

Nx = 64
Ny = 64
NxNy = Nx * Ny
dx = 1.0
dy = 1.0
nstep = 20000
nprint = 50
dtime = 5.0e-2
ttime = 0.0
coefA = 1.0
c0 = 0.40
mobility = 1.0
grad_coef = 1
noise = 0.1

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
ei0 = np.array([
    [0.0567, 0.0168959163898733],
    [0.0168959163898733, 0.016857698027364],
])

v_or = 8.60 * 13.2  * 7.18 * np.sin(116 / 180 * np.pi)
v_ab = 8.15 * 12.85 * 7.12 * np.sin(116 / 180 * np.pi)

# Cij[GPa] * 10^9 * v[Å] * 10*(-30) * NA[/mol] = [/mol]
# [Pa/J] = [1/L3]
cp = np.array([
    [93.9, 52.2, -26.2],
    [  0,  82.1, -19.5],
    [  0,     0,  44.2]
]) * 10**9 / (R * T) * v_ab * 10**(-30) * 6.02 * 10**23 / 4

cm = np.array([
    [93.9, 52.2, -26.2],
    [  0,  82.1, -19.5],
    [  0,     0,  44.2]
]) * 10**9 / (R * T) * v_or * 10**(-30) * 6.02 * 10**23 / 4

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

# ei011 = ei0[0,0]
# ei012 = ei0[0,1]
# ei022 = ei0[1,1]

# cp11 * ei011 + cp12 * ei022 + 0 * ei012
# 0 * ei011 + cp11 * ei022 + 0 * ei012
# 0 * ei011 + 0    * ei022 + cp44 * ei012
#%%

# applied strains
ea = np.array([0.00, 0.00, 0.00])

# initialize stress
s11 = np.zeros((Nx, Ny))
s22 = np.zeros((Nx, Ny))
s12 = np.zeros((Nx, Ny))

# initialize strain
e11 = np.zeros((Nx, Ny))
e22 = np.zeros((Nx, Ny))
e12 = np.zeros((Nx, Ny))
# %%
# con = micro_ch_pre(Nx, Ny, c0, noise)
con = np.load('../data/con1.npy')
# %%
bulk = np.mean(con)

# %%
kx, ky, k2, k4 = prepare_fft(Nx, Ny, dx, dy)
# %%
tmatx = green_tensor(Nx,Ny,kx,ky,cp,cm)

# %%

for istep in range(1, nstep + 1):
    ttime += dtime
    
    # Calculate derivatives of free energy and elastic energy
    delsdc, et11, et22, et12, s11, s22, s12, el = solve_elasticity_v2(Nx,Ny,tmatx,cm,cp,ea,ei0,con)

    # print(delsdc)
    # raise TypeError()

    # Assuming you have the free_energ_ch_v2 and solve_elasticity_v2 functions
    dfdcon, g = free_energ_ch_v2(Nx,Ny,con,w)

    energy_g[istep-1] = np.sum(g)
    energy_el[istep-1] = np.sum(el)
    
    conk = fft2(con)
    dfdconk = fft2(dfdcon)
    delsdck = fft2(delsdc)
    
    # Time integration
    numer = dtime * mobility * k2 * (dfdconk + delsdck)
    denom = 1.0 + dtime * coefA * mobility * grad_coef * k4
    conk = (conk - numer) / denom
    con = np.real(ifft2(conk))
    
    # Clip small deviations
    # con = np.clip(con, 0.00001, 0.9999)
    # print(np.mean(con))
    con = con * (1 + np.sum(con[con < 0])/bulk)
    con = con * (1 + np.sum(con[con > 1])/bulk)
    con[con < 0] = 0
    con[con > 1] = 1
    # print("----------")
    # print(bulk)
    # print(np.sum(con)/ len(con))
    # # バルク規格化
    # print("----------")
    # con = bulk - np.sum(con) / (Nx * Ny) + con
    # con[0,0] = 0
    # con[Nx-1,Nx-1] = 1
    
    # Print results
    # if ((np.mean(con)/bulk <0.99)):
    #     raise TypeError()

    if (istep % nprint == 0) or (istep == 1) or (np.mean(con)/bulk <0.99):
        # plt.imshow(con_disp)の図の向きは、
        # y
        # ↑
        # |
        # + --→ x [100]
        # となる。
        con_disp = np.flipud(con.transpose())
        myplt.get_matrix_image(con_disp)


# Calculate compute time
compute_time = time.process_time() - time0
print(f"Compute Time: {compute_time:.10f}")

#%%