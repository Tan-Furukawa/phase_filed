#%%
class SampleIterator(object):
    def __init__(self):
        self.num = 3
        self.current = 0
        self.__test = 0

    def __iter__(self):
        print("iter")
        print(self.__test)
        print("iter")
        return self

    def __next__(self):
        print("next")
        print(self.current)
        if self.current == self.num:
            raise StopIteration()

        ret = self.current
        self.current += 1
        return ret

itr = SampleIterator()
itr.num = 3
for i in itr:
    print ("---------------")

k = lambda x, y: x + y
print(k(1,4))
# %%
