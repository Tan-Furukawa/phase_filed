#%%
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
c0 = 0.45
mobility = 1.0
grad_coef = 0.4

energy_g = np.zeros(nstep) + np.nan
energy_el = np.zeros(nstep) + np.nan

# coefA * c * (1-c) + grad_coef * (∇c)^2

# matrix
# cm11 = 1400.0
# cm12 = 600.0
# cm44 = 400.0

R = 8.31446262
T = 773

Fe = 55.845
rho_Fe = 7.874 * 10 ** 6

Mo = 95.95 
rho_Mo = 10.28 * 10 ** 6
# w = 36490 / (R * T)

w = 3
# x = np.linspace(0.001, 0.999, 100)
# plt.plot(x * np.log(x) + (1-x) * np.log(1-x) + w * x * (1-x))

a = (0.286 * 10**(-9)) ** 3 * 6.02 * 10**23 / 2
cp11 = 4.63 * 10**11 / (R * T) * (a)
cp12 = 1.61 * 10**11 / (R * T) * (a)
cp44 = 1.09 * 10**11 / (R * T) * (a)

cm11 = 2.33 * 10**11 / (R * T) * (a)
cm12 = 1.35 * 10**11 / (R * T) * (a)
cm44 = 1.18 * 10**11 / (R * T) * (a)

# cp11 = 4.63 * 10**11 / (R * T) * (Mo/rho_Mo)
# cp12 = 1.61 * 10**11 / (R * T) * (Mo/rho_Mo)
# cp44 = 1.09 * 10**11 / (R * T) * (Mo/rho_Mo)
# cm11 = 2.33 * 10**11 / (R * T) * (Fe/rho_Fe)
# cm12 = 1.35 * 10**11 / (R * T) * (Fe/rho_Fe)
# cm44 = 1.18 * 10**11 / (R * T) * (Fe/rho_Fe)

# cm11 = 2.33 * 10**11 / (R * T) * (Fe/rho_Fe)
# cm12 = 1.35 * 10**11 / (R * T) * (Fe/rho_Fe)
# cm44 = 1.18 * 10**11 / (R * T) * (Fe/rho_Fe)
# cp11 = cm11 
# cp12 = cm12 
# cp44 = cm44 

#%%
# eigen strains 
# ei0_ij * c(r)
ei0 = 0.05
noise = 0.3

# strain components due to lattice defects
# zero tensor
# ed11 = np.zeros((Nx, Ny))
# ed22 = np.zeros((Nx, Ny))
# ed12 = np.zeros((Nx, Ny))

# applied strains
ea = np.array([0.00, 0.00, 0.0])

# initialize stress
s11 = np.zeros((Nx, Ny))
s22 = np.zeros((Nx, Ny))
s12 = np.zeros((Nx, Ny))

# initialize strain
e11 = np.zeros((Nx, Ny))
e22 = np.zeros((Nx, Ny))
e12 = np.zeros((Nx, Ny))
# %%
con = micro_ch_pre(Nx, Ny, c0, noise)
bulk = np.sum(con) / (Nx * Ny)

# %%
kx, ky, k2, k4 = prepare_fft(Nx, Ny, dx, dy)
# %%
tmatx = green_tensor(Nx,Ny,kx,ky,cm11,cm12,cm44,cp11,cp12,cp44)

# %%

for istep in range(1, nstep + 1):
    ttime += dtime
    
    # Calculate derivatives of free energy and elastic energy
    delsdc, et11, et22, et12, s11, s22, s12, el = solve_elasticity_v2(Nx,Ny,tmatx,cm11,cm12,cm44,cp11,cp12,cp44,ea,ei0,con)
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
    con = np.clip(con, 0.00001, 0.9999)
    # print("----------")
    # print(bulk)
    # print(np.sum(con)/ len(con))
    # # バルク規格化
    # print("----------")
    con = bulk - np.sum(con) / (Nx * Ny) + con
    # con[0,0] = 0
    # con[Nx-1,Nx-1] = 1
    
    # Print results
    if (istep % nprint == 0) or (istep == 1):
        # plt.plot(energy_el,label='el')
        # plt.plot(energy_g, label='g')
        # plt.plot(energy_g + energy_el, label = "bulk")
        # plt.legend()
        # plt.show()
        # print(f"done step: {istep}")
        # plt.imshow(el, cmap='plasma', interpolation='nearest')
        # plt.colorbar()  # カラーバーを追加
        # plt.title('energy')
        # plt.show()
        plt.imshow(con, cmap='Greys', interpolation='nearest')
        # plt.imshow(s11+s22+s12, cmap='viridis', interpolation='nearest')
        plt.colorbar()  # カラーバーを追加
        plt.title('concentration')
        plt.show()

        # Write results to files or perform other output actions

# Calculate compute time
compute_time = time.process_time() - time0
print(f"Compute Time: {compute_time:.10f}")

#%%