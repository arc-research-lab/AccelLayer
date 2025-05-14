import numpy as np
import math
import sys
from CDAC import *


def main():


    DATA_TYPE = 4
    num_acc = 1

    # LINEAR_MODEL_IN=np.array([
    # [3072,1024,3072,1], 
    # [512,64,512,96],  
    # [512,512,64,96],    
    # [3072,1024,1024,1],  
    # [3072,1024,4096,1],
    # [3072,4096,1024,1],
    # ])

    # Read parameters from terminal.
    seq = int(input("Enter sequence length (seq): "))
    batch = int(input("Enter batch size (batch): "))
    head_dim = int(input("Enter head dimension (head_dim): "))
    heads = int(input("Enter number of heads (heads): "))
    mlp_ratio = int(input("Enter MLP ratio (mlp_ratio): "))

    embed_dim = heads * head_dim
    mlp_dim = embed_dim * mlp_ratio

    # Generate linear workload array.
    LINEAR_MODEL_IN = np.array([
        [seq * batch, embed_dim, embed_dim * 3, 1],
        [seq * batch, head_dim, seq, heads],
        [seq * batch, seq, head_dim, heads],
        [seq * batch, embed_dim, embed_dim, 1],
        [seq * batch, embed_dim, mlp_dim, 1],
        [seq * batch, mlp_dim, embed_dim, 1],
    ])

    # Generate non-linear workload array.
    transpose0_datasize = 3 * (seq * batch) * embed_dim * DATA_TYPE
    softmax_datasize = (seq * batch) * seq * heads * DATA_TYPE
    transpose1_datasize = (seq * batch) * embed_dim * DATA_TYPE
    add0_datasize = (seq * batch) * embed_dim * DATA_TYPE
    layernorm0_datasize = (seq * batch) * embed_dim * DATA_TYPE
    gelu_datasize = (seq * batch) * mlp_dim * DATA_TYPE
    add1_datasize = (seq * batch) * embed_dim * DATA_TYPE
    layernorm1_datasize = (seq * batch) * embed_dim * DATA_TYPE

    # The off-chip memory bandwidth is set to 8 GB/s.
    offchip_bandwidth = 8
    # ms
    transpose0_time = transpose0_datasize / offchip_bandwidth / 1000000
    softmax_time = softmax_datasize / offchip_bandwidth / 1000000
    transpose1_time = transpose1_datasize / offchip_bandwidth / 1000000
    add0_time = 2 * add0_datasize / offchip_bandwidth / 1000000
    layernorm0_time = layernorm0_datasize / offchip_bandwidth / 1000000
    gelu_time = gelu_datasize / offchip_bandwidth / 1000000
    add1_time = 2 * add1_datasize / offchip_bandwidth / 1000000
    layernorm1_time = layernorm1_datasize / offchip_bandwidth / 1000000

    
    print("\n LINEAR_MODEL_IN =")
    print(LINEAR_MODEL_IN)

    

    part_final, final_config, layer_cycle = cdac_top(LINEAR_MODEL_IN,DATA_TYPE,num_acc)


    total_time = layer_cycle[0]/1000000 + transpose0_time + layer_cycle[1]/1000000 + softmax_time + layer_cycle[2]/1000000 + transpose1_time + \
                layer_cycle[3]/1000000 + add0_time + layernorm0_time + layer_cycle[4]/1000000 + gelu_time + layer_cycle[5]/1000000 + add1_time + \
                layernorm1_time

    print("============================")
    print("============================")
    print("The latency for each layer is below:")
    print("QKV gen MM: {}ms.".format(layer_cycle[0]/1000000))
    print("QKV transpose layer: {}ms.".format(transpose0_time))
    print("Q*K batch MM: {}ms.".format(layer_cycle[1]/1000000))
    print("Softmax layer: {}ms.".format(softmax_time))
    print("K*V batch MM: {}ms.".format(layer_cycle[2]/1000000))
    print("Transpose layer: {}ms.".format(transpose1_time))
    print("Projection MM: {}ms.".format(layer_cycle[3]/1000000))
    print("Add layer: {}ms.".format(add0_time))
    print("LayerNorm layer: {}ms.".format(layernorm0_time))
    print("FC MM: {}ms.".format(layer_cycle[4]/1000000))
    print("GeLU layer: {}ms.".format(gelu_time))
    print("FC MM: {}ms.".format(layer_cycle[5]/1000000))
    print("Add layer: {}ms.".format(add1_time))
    print("LayerNorm layer: {}ms.".format(layernorm1_time))
    print("============================")
    print("Total Transformer Block: {}ms.".format(total_time))
    print("============================")
    print("============================")









if __name__ == "__main__":
    main()