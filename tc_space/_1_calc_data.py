#%%
# manually calculate phase field data for T, c space
# it takes much time for computation

from binary_monoclinic_3d import phase_field
from matrix_plot import dim3_plot
import numpy as np
import matplotlib.pylab as plt
import cupy as cp

# (temperature, concentration)
parameters = \
    [(t, 0.1) for t in 100 + np.linspace(0, 100, 11) + 273]\
    # [(t, 0.30) for t in 450 + np.linspace(0, 100, 11) + 273] + 
    # [(t, 0.35) for t in 450 + np.linspace(0, 100, 11) + 273] +
    # [(t, 0.7) for t in 200 + np.linspace(0, 100, 11) + 273]
    # [(t, 0.45) for t in 400 + np.linspace(0, 100, 11) + 273] +
    # [(t, 0.55) for t in 350 + np.linspace(0, 100, 11) + 273]
    # [(t, 0.4) for t in 400 + np.linspace(0, 200, 21) + 273] +
    # [(t, 0.5) for t in 300 + np.linspace(0, 200, 21) + 273] +
    # [(t, 0.6) for t in 200 + np.linspace(0, 200, 21) + 273] +
    # [(t, 0.8) for t in 100 + np.linspace(0, 200, 21) + 273]

for (t, c) in parameters:
    try:
        feldspar = phase_field.BinaryMonoclinic3D("tc_space_result")
        feldspar.make_save_file()
        feldspar.c0 = c
        feldspar.T = t
        feldspar.nprint = None
        feldspar.nstep = 400
        feldspar.doit()
    except:
        save_path = f"{feldspar.save_path}/{feldspar.dirname}"
        with open(f"{save_path}/failed.txt", "w") as file:
            file.write("failed computation")

# %%

# feldspar = phase_field.BinaryMonoclinic3D("tc_space_result")
# res = feldspar.load_instance(
# save_path="tc_space_result",
# dir_name="output_2023-12-19-13-38-23"
# )
# print(res)

# dat = np.load("tc_space_result/output_2023-12-19-13-02-05/res/con_400.npy")
# dim3_plot.display_3d_matrix(dat)

# %%
