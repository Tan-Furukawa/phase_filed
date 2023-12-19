#%%
import json
import numpy as np
import datetime
import os
import shutil

def create_directory(directory_name="tmp", force=False):
    print(f"making <{directory_name}> directory...")
    if os.path.exists(directory_name):
        if force:
            shutil.rmtree(directory_name)
            print(f"Removed existing directory: {directory_name}")
        else:
            print(f"The directory <{directory_name}> already exists.")
            return

    os.makedirs(directory_name)
    print(f"Created a new directory: {directory_name}")

def kwargs_to_json(**kwargs):
    def convert_to_serializable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()  # NumPy配列をリストに変換
        else:
            return value

    kwargs_serializable = {key: convert_to_serializable(value) for key, value in kwargs.items()}
    return json.dumps(kwargs_serializable)

def save_json(path, json):
  with open(path, 'w') as file:
    file.write(json)

def make_dir_name(file_name="output"):
  current_time = datetime.datetime.now()
  timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")
  filename = f"{file_name}_{timestamp}"
  return filename

def get_base_name(str):
    basename = os.path.basename(str)
    return basename.split(".")[0]



if __name__ == "__main__":
  create_directory("tmp")
  dirname = make_dir_name()
  create_directory(f"tmp/{dirname}", force=True)
  kwargs = {
      'arr': np.array([[1,2,4],[2,3,4]]),
      'name': 'Alice',
      'age': 30,
      'city': 'New York'
  }
  result = kwargs_to_json(**kwargs)
  save_json(f"tmp/{dirname}/_parameters.json", result)
  print(result)

  print(get_base_name('result/output_2023-12-06-19-43-56/con_3330.npy'))


#%%