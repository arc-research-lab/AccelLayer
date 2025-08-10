import numpy as np
from ea_algorithm_block import *
from draw_pipeline import *
from mem_schedule import *
from inter_schedule import *
import sys

#################################################################################

num_batch=1
num_acc=4

DATA_TYPE=1 ## INT8

num_block=int(1)
MODEL_BLOCK = np.array([
        # embeded:
        [128, 17, 128, 1, 0],
        [128, 128, 512, 1, 0],
        [128, 512, 128, 1, 0],
        # block:
        [128, 128, 128, 1, 0],
        [128, 128, 128, 1, 0],
        [128, 128, 128, 1, 0],
        [128, 128, 128, 1, 0],
        [128, 128, 128, 1, 0],
        [128, 128, 512, 1, 0],
        [128, 512, 128, 1, 0],
        [128, 128, 128, 1, 0],
        # class block:
        [1, 128, 128, 1, 0],
        [129, 128, 128, 1, 0],
        [129, 128, 128, 1, 0],
        [1, 128, 129, 1, 0],
        [1, 129, 128, 1, 0],
        [1, 128, 128, 1, 0],
        [1, 128, 512, 1, 0],
        [1, 512, 128, 1, 0],
    ])

#################################################################################


## Hardware platform information
board_series = 'Versal'
term=38
DDR_BANK=1/num_acc
AIE_NUM=364
PLIO_IN=156
PLIO_OUT=117
BRAM=(967-100) #100 for AXI bound consumpssion
URAM=(463-43)
HW_Part=[DDR_BANK,AIE_NUM,PLIO_IN,PLIO_OUT,BRAM,URAM]
mono_batch=1

num_node=MODEL_BLOCK.shape[0]
num_layer=num_node*num_block
nVar=num_batch*num_layer
nPop=2 #nuber of parents
children_ratio=1 #percentage between children and parents
nChild=round(children_ratio * nPop/2) * 2 
nIter=1
beta=1 # Hyperparameter for selecting the parents 
mutate_ratio=0.1


MODEL_IN=np.zeros([num_layer,5])
for blk in range(0,num_block):
    index=blk*num_node
    MODEL_IN[index:index+num_node,:]=MODEL_BLOCK

depend_per_block_0 = 7
depend_per_block_1 = depend_per_block_0 + 3

num_depend=depend_per_block_0+(num_block-1)*depend_per_block_1
depend_map=np.zeros([num_depend,3]).astype(int)
depend_map[0,:]=[0,3,0] #node_pre, node_nxt, op{-1,0,2,1}: -1 for broadcast
depend_map[1,:]=[1,3,2]
depend_map[2,:]=[2,4,1]
depend_map[3,:]=[3,4,0]
depend_map[4,:]=[4,5,0]
depend_map[5,:]=[5,6,0]
depend_map[6,:]=[6,7,0]

if num_block>1:
    depend_map[7,:]=[7,8,-2]
    depend_map[8,:]=[7,9,-1]
    depend_map[9,:]=[7,10,-1]
    depend_map[depend_per_block_1:(depend_per_block_1+depend_per_block_0),0:2]=np.add(depend_map[0:depend_per_block_0,0:2],num_node)
    depend_map[depend_per_block_1:(depend_per_block_1+depend_per_block_0),2]=depend_map[0:depend_per_block_0,2]

for blk in range(1,num_block-1,1):
    index=depend_per_block_0+blk*depend_per_block_1
    depend_map[index:index+depend_per_block_1,0:2]=np.add(depend_map[index-depend_per_block_1:index,0:2],num_node)
    depend_map[index:index+depend_per_block_1,2]=depend_map[index-depend_per_block_1:index,2]

final_config,final_time_table,final_throughput,final_schedule,final_mem_move,best_cost,best_pos,bestcost_iter,best_layer_cycle=evolution_search(MODEL_IN,HW_Part,DATA_TYPE,num_acc,num_batch,num_node,num_block,depend_map,nPop,nChild,nVar,beta,mutate_ratio,nIter,term,board_series)
if final_config[0,0] != 1e30:
    final_config=final_config.astype(int)


view_config = np.zeros([final_config.shape[0],18])
view_config[:,0:6] = final_config[:,0:6].copy()
view_config[:,6:9] = final_config[:,10:13].copy()
view_config[:,9:13] = final_config[:,22:26].copy()
view_config[:,13:18] = final_config[:,31:36].copy()
print('\n\nHardware Configuration is\n h1, w1, w2, A, B, C, X, Y, Z, part_aL, part_bL, part_bR, part_cR, dup, atten_flag, harden_aie, BRAM, URAM')
print(view_config)
print(best_layer_cycle[0:num_node])
print(best_pos[0:num_node])


print('\n\nBatch = ' + str(num_batch) + ', Acc = ' + str(num_acc) + ', Mono_batch = ' + str(mono_batch) + ', Latency(ns) = ' + str(best_cost) + ', Throughput(GOPS) = ' + str(final_throughput))