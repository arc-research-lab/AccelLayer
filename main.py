import numpy as np
import math
import sys
from CDAC import *

def main():

    MODEL_IN=np.array([
    [3072,1024,3072,1], 
    [512,64,512,96],  
    [512,512,64,96],    
    [3072,1024,1024,1],  
    [3072,1024,4096,1],
    [3072,4096,1024,1],
    ])

    # # Read parameters from terminal.
    # seq = int(input("Enter sequence length (seq): "))
    # batch = int(input("Enter batch size (batch): "))
    # head_dim = int(input("Enter head dimension (head_dim): "))
    # heads = int(input("Enter number of heads (heads): "))
    # mlp_ratio = int(input("Enter MLP ratio (mlp_ratio): "))

    # embed_dim = heads * head_dim
    # mlp_dim = embed_dim * mlp_ratio

    # # Generate workload array.
    # MODEL_IN = np.array([
    #     [seq * batch, embed_dim, embed_dim * 3, 1],
    #     [seq * batch, head_dim, seq, heads],
    #     [seq * batch, seq, head_dim, heads],
    #     [seq * batch, embed_dim, embed_dim, 1],
    #     [seq * batch, embed_dim, mlp_dim, 1],
    #     [seq * batch, mlp_dim, embed_dim, 1],
    # ])


    print("\nMODEL_IN =")
    print(MODEL_IN)

    DATA_TYPE = 4
    num_acc = 2

    part_final, final_config, layer_cycle = cdac_top(MODEL_IN,DATA_TYPE,num_acc)
    




if __name__ == "__main__":
    main()