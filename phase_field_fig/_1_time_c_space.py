#%%
from binary_monoclinic_3d import phase_field
from matrix_plot import dim3_plot, dim2_plot
import numpy as np
import matplotlib.pylab as plt
import cupy as cp
import os
import glob
import pandas as pd
from tqdm import tqdm

def numerical_sort(value):
    value = os.path.basename(value)
    parts = value.split('_')  # ファイル名をアンダースコアで分割
    if len(parts) == 2:
        try:
            return int(parts[1].split('.')[0])  # 数字部分を抜き出して整数に変換
        except ValueError:
            return value
    return value

def get_files_from_dir(dir_path):
    file_pattern = os.path.join(dir_path, '*con*')
    file_list = glob.glob(file_pattern)
    sorted_file_list = sorted(file_list, key=numerical_sort)
    return sorted_file_list

def get_pickle_information(dir_path):
    feldspar = phase_field.BinaryMonoclinic3D("result",method="nonlinear")
    param = feldspar.load_instance(dir_path)
    return param

def make_time_c_space_plot_data(dir_path):
    files = get_files_from_dir(f"{dir_path}/res")
    sampleN = 1000
    save_info = np.zeros((len(files),sampleN))
    for i, file in enumerate(files):
        dat = np.load(file)
        save_info[i,:] = np.random.choice(dat.reshape(1,len(dat)**3)[0], sampleN)
    np.save("time_c_space.npy", save_info)

def get_max_k(dat, kx, kz):
    Nx, Ny, Nz = dat.shape
    datk = cp.fft.fftn(cp.asarray(dat))
    datk[0,0,0] = 0
    datk[0,int(Nx/2),0] = 0
    datkr = cp.real(datk[:,0,:])

    # dat_1dim = datkr.reshape(1,len(datkr)**2)[0]
    # dat_1dim[res] = cp.argmax(cp.abs(datkr))

    max_index = cp.argmax(cp.abs(datkr))

    res = cp.unravel_index(max_index, datkr.shape)
    lambda_x = 2 * cp.pi / cp.abs(kx[res[0]])
    lambda_y = 2 * cp.pi / cp.abs(kz[res[1]])
    print("lambda is:------")
    print(lambda_x)
    print(lambda_y)
    print("----------------")

    return 1.0 / (1.0 / lambda_x + 1.0 / lambda_y)

def get_lambda_from_one_dim_array(x):
    Nx = len(x)
    sign_x = x > 0
    sign_changed_at = np.diff(np.array(sign_x.tolist(),dtype=np.int64))
    array_lambda = np.sum(sign_changed_at == 1)
    if (array_lambda == 0):
        return Nx
    else:
        return Nx / array_lambda

def get_lambda_from_3_dim_mat(mat):
    x = (mat)[:,0,0]
    xl = get_lambda_from_one_dim_array(x)
    z = (mat)[0,0,:]
    zl = get_lambda_from_one_dim_array(z)

    return xl * zl / (xl + zl)

if __name__ == "__main__":
    # dir_path = "result/important"
    # dir_path = "result/output_2023-12-23-11-53-37"
    dir_path = "result/output_2023-12-23-16-50-53"

    #-----------------------------------------------------------------
    # dtime
    #-----------------------------------------------------------------
    param = get_pickle_information(dir_path)
    print(param)
    print("dtime is:")
    print(param["dtime"])
    print("nsave is:")
    print(param["nsave"])
    #-----------------------------------------------------------------
    # make time_c_space plot
    #-----------------------------------------------------------------
    # make_time_c_space_plot_data(dir_path)


    #-----------------------------------------------------------------
    # dim3_plot.display_3d_matrix(x)
    #-----------------------------------------------------------------
    param = get_pickle_information(dir_path)
    files = get_files_from_dir(f"{dir_path}/res")
    file = files[100]
    lambda_result = np.zeros(len(files))
    Nx = Ny = Nz = 64
    dx = dy = dz = 1
    kx, ky, kz, k2, k4,_,_,_ = phase_field.prepare_fft(Nx, Ny, Nz, dx, dy, dz)
    for i, file in tqdm(enumerate(files)):
        mat = np.load(file) - param["c0"]
        lambda_result[i] = get_max_k(mat, kx, kz)
        # get_lambda_from_3_dim_mat(mat)

    np.save("lambda_result.npy" ,lambda_result)
    #%%





# %%
