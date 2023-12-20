import cupy as np

def add_initial_noise(Nx, Ny, Nz, c0, noise):

  con = np.zeros((Nx, Ny, Nz))

  rng = np.random.default_rng(seed=123)
  con = c0 + noise * (0.5 - rng.random((Nx, Ny, Nz)))
  # np.random.seed(123)
  # con = c0 + noise * (0.5 - np.random.rand(Nx, Ny))

  return con
