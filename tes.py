# # import numpy as np
#%%
import multiprocessing
def worker_function_1():
    for i in range(5):
        print(f"Worker 1: Task {i}")
        # 何らかの非同期処理を実行
    print("Worker 1 finished")

def worker_function_2():
    for i in range(5):
        print(f"Worker 2: Task {i}")
        # 何らかの非同期処理を実行
    print("Worker 2 finished")

# if __name__ == "__main__":
# プロセスを作成
process1 = multiprocessing.Process(target=worker_function_1)
process2 = multiprocessing.Process(target=worker_function_2)

# プロセスを開始
process1.start()
process2.start()

# プロセスが終了するまで待機
process1.join()
process2.join()

print("Main program finished")