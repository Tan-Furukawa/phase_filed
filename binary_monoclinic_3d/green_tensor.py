import numpy as np

def green_tensor (Nx, Ny, Nz, kx, ky, kz, cp, cm):

  # cubic elastic constant
  # C = 
  # c11 c12  0 
  # c12 c11  0 
  #  0   0  c44

  c = 0.5 * (cm + cp)

  omeg11 = np.zeros((Nx, Ny, Nz))
  omeg12 = np.zeros((Nx, Ny, Nz))
  omeg13 = np.zeros((Nx, Ny, Nz))
  omeg22 = np.zeros((Nx, Ny, Nz))
  omeg23 = np.zeros((Nx, Ny, Nz))
  omeg33 = np.zeros((Nx, Ny, Nz))

  for i in range(Nx):
    for j in range(Ny):
      for l in range(Nz):
        # if(rr < 1.0e-8): #これでいいの？
        #   d0=1.0
        a11 = c[0, 0]*kx[i]**2 + 2*c[0, 4]*kx[i]*kz[l] + c[4, 4]*kz[l]**2 + c[5, 5]*ky[j]**2
        a12 = c[0, 1]*kx[i]*ky[j] + c[1, 4]*ky[j]*kz[l] + c[3, 5]*ky[j]*kz[l] + c[5, 5]*kx[i]*ky[j]
        a13 = c[0, 2]*kx[i]*kz[l] + c[0, 4]*kx[i]**2 + c[2, 4]*kz[l]**2 + c[3, 5]*ky[j]**2 + c[4, 4]*kx[i]*kz[l]
        a22 = c[1, 1]*ky[j]**2 + c[3, 3]*kz[l]**2 + 2*c[3, 5]*kx[i]*kz[l] + c[5, 5]*kx[i]**2
        # a22 = 1.0
        a23 = c[1, 2]*ky[j]*kz[l] + c[1, 4]*kx[i]*ky[j] + c[3, 3]*ky[j]*kz[l] + c[3, 5]*kx[i]*ky[j]
        a33 = c[2, 2]*kz[l]**2 + 2*c[2, 4]*kx[i]*kz[l] + c[3, 3]*ky[j]**2 + c[4, 4]*kx[i]**2
        det = a11*a22*a33 - a11*a23*a23 - a12*a12*a33 + a12*a23*a13 + a13*a12*a23 - a13*a22*a13

        if(det == 0): #これでいいの？
          omeg11[i,j,l] = 0
          omeg12[i,j,l] = 0
          omeg13[i,j,l] = 0
          omeg22[i,j,l] = 0
          omeg23[i,j,l] = 0
          omeg33[i,j,l] = 0
        else:
          # omeg_ik = (C_ijkl * k_j * k_l)^-1
          omeg11[i,j,l] = (a22*a33 - a23**2) / det
          omeg12[i,j,l] = (-a12*a33 + a13*a23) / det
          omeg13[i,j,l] = (a12*a23 - a13*a22) / det
          omeg22[i,j,l] = (a11*a33 - a13**2) / det
          omeg23[i,j,l] = (-a11*a23 + a12*a13) / det
          omeg33[i,j,l] = (a11*a22 - a12**2) / det

  
  # print(omeg22)
  tmatx = np.zeros((Nx, Ny, Nz, 3, 3, 3, 3))

  # gmatx:
  # 1/2 * (k_k * G_pl + k_l * G_pk) * k_q
  for i in range(Nx):
    for j in range(Ny):
      for l in range(Ny):
          # Greens tensor
          gmatx = np.zeros((3, 3))
          gmatx[0, 0] = omeg11[i, j, l]
          gmatx[0, 1] = omeg12[i, j, l]
          gmatx[0, 2] = omeg13[i, j, l]
          gmatx[1, 0] = omeg12[i, j, l]
          gmatx[1, 1] = omeg22[i, j, l]
          gmatx[1, 2] = omeg23[i, j, l]
          gmatx[2, 0] = omeg13[i, j, l]
          gmatx[2, 1] = omeg23[i, j, l]
          gmatx[2, 2] = omeg33[i, j, l]
          
          # position vector
          dvect = np.zeros(3)
          dvect[0] = kx[i]
          dvect[1] = ky[j]
          dvect[2] = kz[l]
          
          # Green operator
          for kk in range(3):
              for ll in range(3):
                  for ii in range(3):
                      for jj in range(3):
                          tmatx[i, j, l, kk, ll, ii, jj] = 0.25 * (
                              gmatx[ll, ii] * dvect[jj] * dvect[kk] +
                              gmatx[kk, ii] * dvect[jj] * dvect[ll] +
                              gmatx[ll, jj] * dvect[ii] * dvect[kk] +
                              gmatx[kk, jj] * dvect[ii] * dvect[ll]
                          )
  return tmatx, omeg11
    
