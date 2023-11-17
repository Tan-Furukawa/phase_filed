# # import numpy as np
#%%
import multiprocessing
import time

# プロセスを作成
# target=lambda q, func: q.put(func()), args=(results, func)
def f1():
    num=0
    for i in range(5000000):
        num += i
        # 何らかの非同期処理を実行
    return num

def f2():
    num=0
    for i in range(5000000):
        num += i
    return num

def target(q, fn):
    return q.put(fn())

def mul():

    results = multiprocessing.Queue()
    processes = []

    for func in [f1,  f2]:
        process = multiprocessing.Process(target=target, args=(results, func))
        processes.append(process)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    results_list = []

    while not results.empty():
        result = results.get()
        results_list.append(result)

    print(results_list)

    print("Main program finished")

if __name__ == "__main__":
    print("----------------")
    start_time = time.time() 
    mul()
    print(time.time() - start_time)

    print("----------------")
    start_time = time.time() 
    f1()
    f2()
    print(time.time() - start_time)
    print("----------------")

#%% 