#%%

# making data for plot

from binary_monoclinic_3d import phase_field
from matrix_plot import dim3_plot
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
import cupy as cp

feldspar = phase_field.BinaryMonoclinic3D("tc_space_result")
save_path = "tc_space_result"
# c0 = 0.1
dir_list = glob.glob(f"{save_path}/*")

feldspar.make_calculation_parameters()
feldspar.prepare_result_variables()

df = pd.DataFrame(columns=['t', 'c0', 'div1', 'div2'])
#%%
for dir in dir_list:
    param = feldspar.load_instance(
        full_dir_path = dir
    )
    dat1 = np.load(f"{dir}/res/con_200.npy")
    dat2 = np.load(f"{dir}/res/con_400.npy")

    div1 = np.max(dat1) - np.min(dat1)
    div2 = np.max(dat2) - np.min(dat2)
    new_row = pd.DataFrame({"t":[param["T"]-273],"c0":[param["c0"]],"div1":[div1], "div2":[div2]})
    df = pd.concat([df, new_row])
    if (param["T"] < 310+273 and param["c0"] == 0.2):
        print(dir)

# %%
df_growth = df[df["div1"] < df["div2"]]
df_growth = df_growth.sort_values("t")
print(df_growth)
df_decline = df[df["div1"] > df["div2"]]
df_decline = df_decline.sort_values("t",ascending=False)

df.to_csv(f"tc_space.csv")
# print(df_decline)
# %%
fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(
    df_growth["c0"],
    df_growth["t"],
    c="darkgrey",
    marker="^",
    edgecolor="black",
    label = "2 phase"
    )
ax.scatter(
    df_decline["c0"],
    df_decline["t"],
    c="white",
    edgecolor="black",
    label = "single phase",
    marker="v"
    )
plt.legend(
    bbox_to_anchor=(0.5, 1),
    loc='upper left',
    borderaxespad=0,
    fontsize=8,
    labelspacing=1,
    prop={'family' : 'arial'}
    )
ax.set_ylim(000,700)
ax.set_xlim(0,1)
plt.savefig("tc_space.pdf", format="pdf")
plt.show()

# %%
