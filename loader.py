import numpy as np
import os




def yield_batch(fdirs):
    while True:
        for fdir in fdirs:
            data = np.load(fdir)
            data,label = data["arr_0"], data["arr_1"]

            yield data,label
