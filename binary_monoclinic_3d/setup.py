#%%
from setuptools import setup, find_packages

setup(
    name='binary_monoclinic_3d',  # パッケージ名（pip listで表示される）
    version="0.0.1",  # バージョン
    description="sample of minimum package",  # 説明
    author='Furukawa Tan',  # 作者名
    packages=find_packages(),  # 使うモジュール一覧を指定する
    license='MIT'  # ライセンス
)

# pip install -e .

# %%
