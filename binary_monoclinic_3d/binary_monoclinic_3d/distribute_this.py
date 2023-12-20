#%%
import shutil
import os

# 現在のディレクトリを取得
current_directory = os.getcwd()
#%%
destination_directory = "../analysis_binary_monoclinic/_binary_monoclinic" 
shutil.copytree(current_directory, destination_directory,dirs_exist_ok=True)

#%%