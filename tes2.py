#%%
import tes as t
import time

if __name__ == "__main__":
    print("--------1-------")
    start_time = time.time() 
    t.mul()
    print(time.time() - start_time)

    print("-------2--------")
    start_time = time.time() 
    t.f1()
    t.f2()
    print(time.time() - start_time)
    print("----------------")