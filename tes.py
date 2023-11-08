#%%
import matplotlib.pyplot as plt
import numpy as np

# データを生成
data = np.random.rand(10, 10)  # 仮のデータ
vmin = 0.2  # 下限
vmax = 0.8  # 上限
norm = plt.Normalize(vmin, vmax)

# カラーマップを設定
cmap = plt.get_cmap('viridis')  # 例として'viridis'を使用

# カラーマップの下限と上限を設定
cmap.set_under('blue')  # 下限を青に設定
cmap.set_over('red')    # 上限を赤に設定

# プロットを作成
plt.imshow(data, cmap=cmap,norm=norm)

# カラーバーを表示
plt.colorbar()

# プロットを表示
plt.show()
#%%