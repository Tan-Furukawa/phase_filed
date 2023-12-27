#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def display_3d_matrix(
      data,
      add_color_bar=True,
      show=True,
      color_map="viridis",
      range=(None, None),
      round_n=1
      ):
  if range[0] is not None:
    data[0,-1,0] = range[0]

  if range[0] is not None:
    data[0,-1,1] = range[1]

  ndata = (data - np.min(data)) / (np.max(data) - np.min(data))
  nx, ny, nz = data.shape

  Yx, Zx = np.mgrid[0:(ny+1), 0:(nz+1)]
  Xx = np.zeros(Yx.shape) + nx

  Zy, Xy = np.mgrid[0:(nz+1), 0:(nx+1)]
  Yy = np.zeros(Zy.shape)

  Xz, Yz = np.mgrid[0:(nx+1), 0:(ny+1)]
  Zz = np.zeros(Xz.shape) + nz

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.set_xlim(0, nx + 1)
  ax.set_ylim(-1, ny)
  ax.set_zlim(0, nz + 1)

  if (color_map == 'BrBG'):
    col_fn = plt.cm.BrBG
  elif (color_map == 'plasma'):
    col_fn = plt.cm.plasma
  elif (color_map == 'viridis'):
    col_fn = plt.cm.viridis
  else:
    raise TypeError()

  surf = ax.plot_surface(Xx, Yx, Zx, rstride=1, cstride=1, facecolors=col_fn(ndata[-1,:,:]), shade=False)
  surf = ax.plot_surface(Xy, Yy, Zy, rstride=1, cstride=1, facecolors=col_fn(ndata[:,0,:].T), shade=False)
  surf = ax.plot_surface(Xz, Yz, Zz, rstride=1, cstride=1, facecolors=col_fn(ndata[:,:,-1]), shade=False)

  if add_color_bar:
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    cbar.set_ticks([0, 1.0])
    cbar.set_ticklabels([round(np.min(data),round_n), round(np.max(data),round_n)])
    plt.cm.BrBG(ndata[-1,:,:])
  if show:
    plt.show()

if __name__ == '__main__':
  data = np.linspace(-5, 10, num=3*4*5).reshape((3, 4, 5))
  display_3d_matrix(
    data,
    show=False,
    add_color_bar=True,
    range=(0,10),
    round_n=2
    )
  # plt.savefig("tmp.jpeg",format="jpeg") OK
  # plt.savefig("tmp.png",format="png") NG
  #%%