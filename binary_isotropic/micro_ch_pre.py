import numpy as np

def micro_ch_pre(Nx, Ny, c0):

  noise = 0.02
  con = np.zeros((Nx, Ny))

  con = c0 + noise * (0.5 - np.random.rand(Nx, Ny))

  return con
