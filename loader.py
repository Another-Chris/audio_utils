import numpy as np
import os

with open("test.npz", "wb") as fhandle: np.savez_compressed(fhandle, np.array([1,2,3]))



def yield_batch(fdirs):
    while True:
        for fdir in fdirs:
            data = np.load(fdir)
            data,label = data["arr_0"], data["arr_1"]

            yield data,label
