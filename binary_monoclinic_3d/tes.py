#%%
import numpy as np
import matplotlib.pyplot as plt
from prepare_fft import prepare_fft
import os
import glob
from _plot import dim3_plot as myplt3

directory_path = 'result/output_2023-11-17-16-05-52'
# directory_path = 'result/output_2023-12-06-19-43-56'

# ワイルドカードを使ってディレクトリ内のファイルを検索
file_pattern_con = os.path.join(f"{directory_path}/res", '*con*')
file_pattern_el = os.path.join(f"{directory_path}/res", '*el*')
file_pattern_s = os.path.join(f"{directory_path}/res", '*s*')
# ファイルリストを取得
file_list_con = glob.glob(file_pattern_con)
file_list_el = glob.glob(file_pattern_el)
file_list_s = glob.glob(file_pattern_s)

def numerical_sort(value):
    value = os.path.basename(value)
    parts = value.split('_')  # ファイル名をアンダースコアで分割
    if len(parts) == 2:
        try:
            return int(parts[1].split('.')[0])  # 数字部分を抜き出して整数に変換
        except ValueError:
            return value
    return value

file_list_con = sorted(file_list_con, key=numerical_sort)
file_list_el = sorted(file_list_el, key=numerical_sort)
file_list_s = sorted(file_list_s, key=numerical_sort)
eigenstrain = np.array([0.0567, 0.0162, 0.016858, 0, 0.016896, 0])

con = np.load(file_list_con[100])
el = np.load(file_list_el[100])

myplt3.display_3d_matrix(el[:,:,:,1])
myplt3.display_3d_matrix(el[:,:,:,1] + con * eigenstrain[1])
#%%
