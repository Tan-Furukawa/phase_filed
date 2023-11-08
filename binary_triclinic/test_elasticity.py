#%%
import numpy as np
# import matplotlib.pyplot as plt
from micro_ch_pre import micro_ch_pre
from prepare_fft import prepare_fft
from green_tensor import green_tensor
from solve_elasticity_v2 import solve_elasticity_v2
from free_energ_ch_v2 import free_energ_ch_v2
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2


Nx = 128
Ny = 128
dx = 1
dy = 1
mat = np.zeros((Nx, Ny))

# 中心座標
center_x = Nx // 2
center_y = Ny // 2

# 半径
radius = 20

# 中心からの距離が半径以下の点を設定
for i in range(Nx):
    for j in range(Ny):
        if (i - center_x) **2 + (j - center_y)**2 <= radius**2:
            mat[i, j] = 1.0

plt.imshow(mat, cmap='viridis', interpolation='nearest')
plt.colorbar()  # カラーバーを追加
plt.show()

# matrix
cm11 = 2330
cm12 = 1350
cm44 = 1180

# cm11 = 1400.0
# cm12 = 600.0
# cm44 = 400.0

# precipitation
cp11 = cm11 * 0.5
cp12 = cm12 * 0.5
cp44 = cm44 * 0.5

ea = np.array([0.0, 0.01, 0.0])
ei0 = 0.05

kx, ky, k2, k4 = prepare_fft(Nx, Ny, dx, dy)
tmatx = green_tensor(Nx,Ny,kx,ky,cm11,cm12,cm44,cp11,cp12,cp44)

plt.imshow(tmatx[:,:,1,1,1,1], cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.show()

_, et11, et22, et12, s11, s22, s12 = solve_elasticity_v2(Nx, Ny, tmatx, cm11, cm12, cm44, cp11, cp12, cp44, ea, ei0, mat)

plt.imshow(et11, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.show()


#%%
