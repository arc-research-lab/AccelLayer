import numpy as np
import math
import sys


def nonLinear(NON_LINEAR_MODEL_IN, BW, DATA_TYPE):

    ele_num = np.prod(NON_LINEAR_MODEL_IN, axis=1)

    result = np.zeros_like(ele_num, dtype=float)
    for i in range(len(ele_num)):
        result[i] = ele_num[i] * DATA_TYPE / 1000000 / BW


    return result
            





