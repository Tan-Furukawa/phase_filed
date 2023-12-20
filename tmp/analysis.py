#%%
import numpy as np
import matplotlib.pyplot as plt
from micro_ch_pre import micro_ch_pre
from prepare_fft import prepare_fft
from _plot import dim3_plot as myplt3
import glob
import os
from _save import save_3d_plot as save

# ディレクトリのパス
# directory_path = 'result/output_2023-11-17-16-05-52/'
directory_path = 'result/output_2023-12-17-16-10-19'

# ワイルドカードを使ってディレクトリ内のファイルを検索
file_pattern = os.path.join(f"{directory_path}/res", '*con*')
# ファイルリストを取得
file_list = glob.glob(file_pattern)
sorted(file_list)

dirname = "pic"
save.create_directory(f"{directory_path}/{dirname}")

#%%
def numerical_sort(value):
    value = os.path.basename(value)
    parts = value.split('_')  # ファイル名をアンダースコアで分割
    if len(parts) == 2:
        try:
            return int(parts[1].split('.')[0])  # 数字部分を抜き出して整数に変換
        except ValueError:
            return value
    return value

sorted_file_list = sorted(file_list, key=numerical_sort)
# numerical_sort(os.path.basename("result/output_2023-11-17-16-05-52/con_1000.npy"))

Nx = Ny = Nz = 64
dx = dy = dz = 1
kx, ky, kz, k2, k4 = prepare_fft(Nx, Ny, Nz, dx, dy, dz)

def get_max_k(dat):

  datk = np.fft.fftn(dat)
  datk[0,0,0] = 0
  datkr = np.real(datk[:,0,:])
#   plt.imshow(datkr)
#   plt.show()
  print(np.argmax(np.abs(datkr)))

  res = np.unravel_index(np.argmax(np.abs(datkr)), datkr.shape)
  lambda_x = 2 * np.pi / np.abs(kx[res[0]])
  lambda_y = 2 * np.pi / np.abs(kz[res[1]])
  return lambda_x, lambda_y

#%%


dat1 = np.load(sorted_file_list[4])
dat1 = dat1[:,0,:]
dat2 = np.load(sorted_file_list[5])
dat2 = dat2[:,0,:]
d1 = dat1 - np.mean(dat1)
d2 = dat2 - np.mean(dat2)
d1k = np.fft.fft(d1)
d2k = np.fft.fft(d2)

plt.imshow(np.real(d2k)  / np.real(d1k))

# np.imag(d2k) / np.imag(d1k) 

plt.imshow(dk)




datk_bef = np.fft.fftn(dat)
datk_bef[0,0] = 0

kkx = np.hstack((kx[int(len(kx)/2):],kx[:int(len(kx)/2)]))
kky = np.hstack((ky[int(len(ky)/2):],ky[:int(len(ky)/2)]))

d11 = datk_bef[:int(len(kkx)/2), :int(len(kky)/2)]
d12 = datk_bef[:int(len(kkx)/2), int(len(kky)/2):]
d21 = datk_bef[int(len(kkx)/2):, :int(len(kky)/2)]
d22 = datk_bef[int(len(kkx)/2):, int(len(kky)/2):]

datk = np.vstack((
  np.hstack((d22, d21)),
  np.hstack((d12, d11))
))

datkr = np.real(np.abs(datk) ** 2)
np.unravel_index(np.argmax(np.abs(datk)), datkr.shape)

plt.imshow(datkr)
plt.colorbar()
plt.show()

res = np.unravel_index(np.argmax(np.abs(datk)), datkr.shape)
lambda_x = 2 * np.pi / np.abs(kkx[res[0]])
lambda_y = 2 * np.pi / np.abs(kky[res[1]])

lambda_x * lambda_y / np.sqrt(lambda_x**2 + lambda_y**2)

#%%
N = len(sorted_file_list)
res = np.zeros((N,2))
for i, file_path in enumerate(sorted_file_list):
  if os.path.isfile(file_path):  # ファイルかどうかを確認
      dat = np.load(file_path)
      base = save.get_base_name(file_path)

      res[i] = list(get_max_k(dat))
      print(res[i])
#%%

y = res[:,0] * res[:,1] / np.sqrt(res[:,0]**2 + res[:,1]**2)
x = np.arange(len(y))
plt.plot(x, y)
# %%
