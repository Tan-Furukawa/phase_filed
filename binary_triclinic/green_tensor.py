import numpy as np

def green_tensor (Nx, Ny, kx, ky, cp, cm):

  # cubic elastic constant
  # C = 
  # c11 c12  0 
  # c12 c11  0 
  #  0   0  c44

  c = 0.5 * (cm + cp)

  omeg11 = np.zeros((Nx, Ny))
  omeg22 = np.zeros((Nx, Ny))
  omeg12 = np.zeros((Nx, Ny))

  for i in range(Nx):
    for j in range(Ny):
      # if(rr < 1.0e-8): #これでいいの？
      #   d0=1.0
      a11 = c[0, 0]*kx[i]**2 + 2*c[0, 2]*kx[i]*ky[j] + c[2, 2]*ky[j]**2
      a12 = c[0, 1]*kx[i]*ky[j] + c[0, 2]*kx[i]**2 + c[1, 2]*ky[j]**2 + c[2, 2]*kx[i]*ky[j]
      a22 = c[1, 1]*ky[j]**2 + 2*c[1, 2]*kx[i]*ky[j] + c[2, 2]*kx[i]**2
      det = a11*a22 - a12**2
      if(det == 0): #これでいいの？
        det=1.0 * c[2, 2]

      # omeg_ik = (C_ijkl * k_j * k_l)^-1
      omeg11[i,j] = a22 / det
      omeg12[i,j] = -a12 / det
      omeg22[i,j] = a11 / det
  
  # print(omeg22)
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
  return tmatx, omeg11
    
