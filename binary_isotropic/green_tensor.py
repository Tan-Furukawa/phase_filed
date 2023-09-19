import numpy as np

def green_tensor (Nx, Ny, kx, ky, cm11, cm12, cm44, cp11, cp12, cp44):

  # cubic elastic constant
  # C = 
  # c11 c12  0   0  
  # c12 c11  0   0 
  #  0   0  c44  0 
  #  0   0   0  c44

  c11 = 0.5 * (cm11 + cp11)
  c12 = 0.5 * (cm12 + cp12)
  # c44 = 0.5 * (cm44 + cp44)
  c44 = 0.5 * (cm44 + cm44)

  chi = (c11 - c12 - 2.0 * c44) / c44

  omeg11 = np.zeros((Nx, Ny))
  omeg22 = np.zeros((Nx, Ny))
  omeg12 = np.zeros((Nx, Ny))
  for i in range(Nx):
    for j in range(Ny):

      rr = kx[i]**2 + ky[j]**2
      d0 = c11 * rr**3 + chi * (c11 + c12) * rr * (kx[i]**2 * ky[j]**2)

      if(rr < 1.0e-8):
        d0=1.0
      

      # omeg_ik = (C_ijkl * k_j * k_l)^-1
      omeg11[i,j] = (c44 * rr**2 + (c11 - c44) * rr * ky[j]**2) / (c44*d0)
      omeg22[i,j] = (c44 * rr**2 + (c11 - c44) * rr * kx[i]**2) / (c44*d0)
      omeg12[i,j] = -(c12 + c44) * kx[i] * ky[j] * rr / (c44*d0)
  
  tmatx = np.zeros((Nx, Ny, 2, 2, 2, 2))

  # gmatx:
  # 1/2 * (k_k * G_pl + k_l * G_pk) * k_q
  for i in range(Nx):
    for j in range(Ny):
        # Greens tensor
        gmatx = np.zeros((2, 2))
        gmatx[0, 0] = omeg11[i, j]
        gmatx[0, 1] = omeg12[i, j]
        gmatx[1, 0] = omeg12[i, j]
        gmatx[1, 1] = omeg22[i, j]
        
        # position vector
        dvect = np.zeros(2)
        dvect[0] = kx[i]
        dvect[1] = ky[j]
        
        # Green operator
        for kk in range(2):
            for ll in range(2):
                for ii in range(2):
                    for jj in range(2):
                        tmatx[i, j, kk, ll, ii, jj] = 0.25 * (
                            gmatx[ll, ii] * dvect[jj] * dvect[kk] +
                            gmatx[kk, ii] * dvect[jj] * dvect[ll] +
                            gmatx[ll, jj] * dvect[ii] * dvect[kk] +
                            gmatx[kk, jj] * dvect[ii] * dvect[ll]
                        )
  return tmatx
    
