#%%
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt


def get_matrix_image(mat, vmin=0, vmax=1, isnorm = True, theme="Greys"):
    cmap = plt.get_cmap(theme) 
    # カラーマップの正規化
    norm = Normalize(vmin=vmin, vmax=vmax)
    # con < 0 のセルを赤色に変更
    cmap.set_under('blue')  # 下限を青に設定
    cmap.set_over('red')    # 上限を赤に設定
    if (isnorm):
      plt.imshow(mat, cmap=cmap, norm=norm)
      plt.colorbar()
      plt.show()
    else:
      plt.imshow(mat, cmap=cmap)
      plt.colorbar()  # カラーバーを追加
      plt.show()

def get_vector_image(mat, vmin=0, vmax=1, isnorm = True, theme="Greys"):
    cmap = plt.get_cmap(theme) 
    # カラーマップの正規化
    norm = Normalize(vmin=vmin, vmax=vmax)
    # con < 0 のセルを赤色に変更
    cmap.set_under('blue')  # 下限を青に設定
    cmap.set_over('red')    # 上限を赤に設定
    if (isnorm):
      plt.imshow(mat, cmap=cmap, norm=norm)
      plt.colorbar()
      plt.show()
    else:
      plt.imshow(mat, cmap=cmap)
      plt.colorbar()  # カラーバーを追加
      plt.show()

def get_matrix_position(matrix):
  rows, cols = matrix.shape
  x_matrix = np.arange(cols)
  x_matrix = np.tile(x_matrix, (rows, 1))
  y_matirx = np.arange(rows).reshape(-1, 1)
  y_matirx = np.tile(y_matirx, (1, cols))

  return x_matrix, y_matirx

def make_partial_True_matrix(repeat_n, mat_size):
  mat = np.full((mat_size,mat_size), False)
  a = np.arange(0, mat_size, repeat_n)
  x_index = np.repeat(a, len(a))
  y_index = np.tile(a, len(a))
  mat[x_index.tolist(), y_index.tolist()] = True
  return mat

def display_gradient(mat, pixel_spacing=3, isnorm=True):
  position_x, position_y = get_matrix_position(mat)
  gradient_y, gradient_x = np.gradient(mat)
  use = make_partial_True_matrix(pixel_spacing, position_x.shape[0])

  # 勾配ベクトルを可視化
  cmap = plt.get_cmap("Greys") 
  # カラーマップの正規化
  if (isnorm):
    norm = Normalize(vmin=-1, vmax=2) #あえて大きめ
    plt.imshow(mat, cmap=cmap, norm=norm)
  else:
    plt.imshow(mat, cmap=cmap)
  plt.quiver(position_x[use], position_y[use], -gradient_x[use], gradient_y[use], scale=10, width=0.005)
  plt.show()

def hist(mat, bins=300):
    plt.hist(mat.flatten(),bins=bins,range=(0,1))
    plt.show()
  #%%