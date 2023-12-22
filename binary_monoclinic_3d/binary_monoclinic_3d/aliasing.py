#%%
import cupy as cp

def get_padding_len(n):
    # return int((n + 3)/4)
    return 0

def add_aliasing(tensor_3d):
    Nx, Ny, Nz = tensor_3d.shape

    padding_len_x = get_padding_len(Nx)
    padding_len_y = get_padding_len(Ny)
    padding_len_z = get_padding_len(Nz)

    area = cp.zeros((
        padding_len_x * 2 + Nx,
        padding_len_y * 2 + Ny,
        padding_len_z * 2 + Nz
    ), dtype=cp.complex128)

    area[padding_len_x:(padding_len_x+Nx),
        padding_len_y:(padding_len_y+Ny),
        padding_len_z:(padding_len_z+Nz)] = tensor_3d

    return area

def remove_aliasing(padding_tensor, shape):
    Nx, Ny, Nz = shape
    padding_len_x = get_padding_len(Nx)
    padding_len_y = get_padding_len(Ny)
    padding_len_z = get_padding_len(Nz)

    return padding_tensor[
    padding_len_x:(padding_len_x+Nx),
    padding_len_y:(padding_len_y+Ny),
    padding_len_z:(padding_len_z+Nz)
    ]


if __name__ == "__main__":
    tensor0 = cp.zeros((4,4,4))+1
    tensor1 = add_aliasing(tensor0)
    print(tensor1)
    tensor2 = remove_aliasing(tensor1, tensor0.shape)
    print(tensor2)

    import numpy as np
    m = np.zeros((4,4)) + 1
#%%