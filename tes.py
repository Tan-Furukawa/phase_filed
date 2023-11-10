#%%
import numpy as np

repeat_n = 2
mat_size = 10

def make_partial_True_matrix(repeat_n, mat_size):
  mat = np.full((mat_size,mat_size), False)
  a = np.arange(0, mat_size, repeat_n)
  x_index = np.repeat(a, len(a))
  y_index = np.tile(a, len(a))
  mat[x_index.tolist(), y_index.tolist()] = True
  return mat