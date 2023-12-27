#%%
from _1_time_c_space import get_files_from_dir, get_pickle_information, get_max_k
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from matrix_plot import dim3_plot, dim2_plot
from binary_monoclinic_3d import phase_field

def fix_matrix_direction(mat):
    return mat.T

if __name__ == "__main__":
    dir_path = "result/important"
    # dir_path = "result/output_2023-12-23-11-53-37"
    # dir_path = "result/output_2023-12-23-16-50-53"
    files = get_files_from_dir(f"{dir_path}/res")
    param = get_pickle_information(dir_path)

    feldspar = phase_field.BinaryMonoclinic3D("result",method="nonlinear")
    feldspar.dtime = 4e-3
    feldspar.nstep = 2
    feldspar.T = param["T"]
    feldspar.c0 = param["c0"]
    feldspar.set_all()
    dat = np.load(files[1000])
    feldspar.con = cp.asarray(dat)
    feldspar.calculate_phase_filed(method="nonlinear")
    Nx = Ny = Nz = param["Nx"]
    dx = dy = dz = 1

    kx, ky, kz, k2, k4,_,_,_ = phase_field.prepare_fft(Nx, Ny, Nz, dx, dy, dz)
    res = get_max_k(dat,kx,kz)

    vec_mat = cp.asnumpy(feldspar.delsdc + feldspar.dfdcon)
    # dim3_plot.display_3d_matrix(vec_mat, round_n=2)
    plt.imshow(fix_matrix_direction(vec_mat[:,1,:]))
    plt.colorbar()

    dim2_plot.display_gradient(
        fix_matrix_direction(vec_mat[:,1,:]),
        isnorm = False,
        scale = 0.05,
        color_map = "viridis",
        color="white",
        show=False
        )
    plt.savefig("chemical_potential.png",format="png", dpi=300)
    # plt.show()
# %%
