#%%
import shutil
import os

# 現在のディレクトリを取得
current_directory = os.getcwd()
#%%
destination_directory = "../binary_triclinic/_plot" 
shutil.copytree(current_directory, destination_directory,dirs_exist_ok=True)

destination_directory = "../binary_isotropic/_plot" 
shutil.copytree(current_directory, destination_directory,dirs_exist_ok=True)
#%%