# # import numpy as np
#%%
import multiprocess as mp
import time
import numpy as np

# プロセスを作成
# target=lambda q, func: q.put(func()), args=(results, func)
def f1():
    num=0
    for i in range(5000000):
        num += i
    return num
    return 1

def f2():
    num=0
    for i in range(5000000):
        num += i
    return num

def target(q, fn):
    q.put(fn())

def mul():

    results = mp.Queue()
    processes = []

    for func in [f1,  f2]:
        process = mp.Process(target=target, args=(results, func))
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
