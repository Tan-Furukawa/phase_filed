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
nstep = 5000
nprint = 50
dtime = 5.0e-2
ttime = 0.0
coefA = 1.0
c0 = 0.4
mobility = 1.0
grad_coef = 0.5

# coefA * c * (1-c) + grad_coef * (∇c)^2

# matrix
# cm11 = 1400.0
# cm12 = 600.0
# cm44 = 400.0

cm11 = 233
cm12 = 135
cm44 = 118

# precipitation
cp11 = cm11 
cp12 = cm12 
cp44 = cm44 

# eigen strains 
# ei0_ij * c(r)
ei0 = 0.05

# strain components due to lattice defects
# zero tensor
# ed11 = np.zeros((Nx, Ny))
# ed22 = np.zeros((Nx, Ny))
# ed12 = np.zeros((Nx, Ny))

# applied strains
ea = np.array([0.00, 0.01, 0.0])

# initialize stress
s11 = np.zeros((Nx, Ny))
s22 = np.zeros((Nx, Ny))
s12 = np.zeros((Nx, Ny))

# initialize strain
e11 = np.zeros((Nx, Ny))
e22 = np.zeros((Nx, Ny))
e12 = np.zeros((Nx, Ny))
# %%
con = micro_ch_pre(Nx, Ny, c0)

# %%
kx, ky, k2, k4 = prepare_fft(Nx, Ny, dx, dy)
# %%
tmatx = green_tensor(Nx,Ny,kx,ky,cm11,cm12,cm44,cp11,cp12,cp44)


# %%

for istep in range(1, nstep + 1):
    ttime += dtime
    
    # Calculate derivatives of free energy and elastic energy
    delsdc, et11, et22, et12, s11, s22, s12 = solve_elasticity_v2(Nx,Ny,tmatx,cm11,cm12,cm44,cp11,cp12,cp44,ea,ei0,con)
    # Assuming you have the free_energ_ch_v2 and solve_elasticity_v2 functions
    dfdcon = free_energ_ch_v2(Nx,Ny,con)
    
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
    
    # Print results
    if (istep % nprint == 0) or (istep == 1):
        print(f"done step: {istep}")
        plt.imshow(s11, cmap='plasma', interpolation='nearest')
        plt.colorbar()  # カラーバーを追加
        plt.title('stress s11')
        plt.show()

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