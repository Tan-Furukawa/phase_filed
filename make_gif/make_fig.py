#%%
import numpy as np
import matplotlib.pyplot as plt
from matrix_plot import dim3_plot
import glob
import os
from PIL import Image

dirname = "output_2023-12-24-21-31-47"
# ------------------------------------------------------------------------
# ファイルの読み込み
# ------------------------------------------------------------------------
# 数字順にファイル名をソートするための関数
def numerical_sort(value):
    value = os.path.basename(value)
    parts = value.split('_')  # ファイル名をアンダースコアで分割
    if len(parts) == 2:
        try:
            return int(parts[1].split('.')[0])  # 数字部分を抜き出して整数に変換
        except ValueError:
            return value
    return value

file_pattern = os.path.join(f"result/{dirname}/res", '*con*')
file_list = glob.glob(file_pattern)
sorted_file_list = sorted(file_list, key=numerical_sort)

#%%
# ------------------------------------------------------------------------
# jpeg画像の保存
# ------------------------------------------------------------------------


#%%

os.mkdir(f"result/{dirname}/pic")
for i, file_path in enumerate(sorted_file_list):
  if i%30 == 0:
    if os.path.isfile(file_path):  # ファイルかどうかを確認
        dat = np.load(file_path)
        base = os.path.basename(file_path).split(".")[0]

        fig = plt.figure(figsize=(6,6))
        dim3_plot.display_3d_matrix(dat, False, False,range=(0.1,0.7))
        plt.savefig(f"result/{dirname}/pic/{base}.png", format="png",  bbox_inches='tight')
        plt.show()
#%%

# ------------------------------------------------------------------------
# gifの作成
# ------------------------------------------------------------------------
file_pattern = os.path.join(f"result/{dirname}/pic", '*con*')
file_list = glob.glob(file_pattern)
sorted_file_list = sorted(file_list, key=numerical_sort)

image_list = []
for i, filename in enumerate(sorted_file_list):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # JPEGファイルを開く
        img = Image.open(filename) 
        image_list.append(img)

output_file = f"result/{dirname}/output.gif"
image_list[0].save(output_file, save_all=True, append_images=image_list[1:], duration=100, loop=0)

#%%
