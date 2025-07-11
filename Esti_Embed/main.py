import numpy as np
import math
import sys
from CDAC import *
from NONL import *


def main():


    DATA_TYPE = 4
    num_acc = 1


    # Generate linear workload array for Block.
    LINEAR_MODEL_IN = np.array([
        [128, 17, 128, 1],
        [128, 128, 512, 1],
        [128, 512, 128, 1],
    ])

    # Generate non-linear workload array for Block.
    NON_LINEAR_MODEL_IN = np.array([
        [17, 128, 1],
        [17, 128, 1],
        [128, 17, 1],
        [128, 128, 1],
        [128, 512, 1],
        [128, 512, 1],
        [128, 128, 1],
    ])
    offchip_bandwidth = 8

    
    print("\n LINEAR_MODEL_IN =")
    print(LINEAR_MODEL_IN)
    print("\n NON_LINEAR_MODEL_IN =")
    print(NON_LINEAR_MODEL_IN)



    _, _, linear_layer_cycle = cdac_top(LINEAR_MODEL_IN,DATA_TYPE,num_acc)
    nonlinear_layer_time = nonLinear(NON_LINEAR_MODEL_IN, offchip_bandwidth, DATA_TYPE)


    total_linear_time = 0
    for i in range(len(linear_layer_cycle)):
        total_linear_time += linear_layer_cycle[i] / 1_000_000
    
    total_nonlinear_time = 0
    for i in range(len(nonlinear_layer_time)):
        total_nonlinear_time += nonlinear_layer_time[i]
    
    total_time = total_linear_time + total_nonlinear_time

    # print(linear_layer_cycle)
    # print(nonlinear_layer_time)
    print("The total execution time for Block is (ms)")
    print(total_time)









if __name__ == "__main__":
    main()