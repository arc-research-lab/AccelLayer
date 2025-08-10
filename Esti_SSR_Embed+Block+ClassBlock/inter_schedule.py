import numpy as np

#The inter_schedule function is a greedy based algorithm to resolve dependecy and return time step consumpssion
def inter_schedule(num_acc,num_layer,parent,depend_map):
    num_batch=parent.shape[0]//num_layer
    time_start=np.zeros((num_batch,num_layer))
    time_end=np.zeros((num_batch,num_layer))
    assign_flag=np.zeros((num_batch,num_layer))  # Record if the layer has already been secheduled
    sche_flag=(np.min(assign_flag)==0)
    
    num_depend=depend_map.shape[0]
    num_feature=depend_map.shape[1]
    depend_table=np.ones([num_batch,num_depend,num_feature])*(-1)# Record unresolved dependency table
    for bat in range(num_batch):
        depend_table[bat,:,:]=depend_map
    
    exe_time=np.ones([num_acc,num_layer]) # Assume each layer consumes one time step
    assign_pool=[]
    for acc in range(num_acc):
        index=np.where(parent==acc)
        assign_pool.append(list(index[0]))

    #print(assign_pool)
    time_count=np.zeros([num_acc])
    exe_temp=np.zeros([num_acc])
    while sche_flag:
        for acc in range(num_acc):
            length=len(assign_pool[acc])
            for i in range(length):
                layer=assign_pool[acc][i]
                bat=layer//num_layer
                node=layer%num_layer
                depend_flag=np.where(depend_table[bat,:,1]==node) #Check if current node is free to be scheduled
                depend_flag=list(depend_flag[0])
                if len(depend_flag)==0 and assign_flag[bat,node]==0:
                    op_index=np.where(depend_map[:,1]==node) #Find if current layer depends on other layers
                    op_index=list(op_index[0])
                    if len(op_index)==0:#When no children layers, the layer can be schedule when acc is available
                        time_temp=time_count[acc]
                    else:
                        depend_index=depend_map[op_index,0] # Get the index of the child layer
                        time_temp=np.max(np.append(time_count[acc],time_end[bat,depend_index]))
                    
                    if  (time_temp - time_count[acc])<1e-3:
                        assign_pool[acc].remove(layer)
                        op_index=np.where(depend_map[:,1]==node) #Find if current layer depends on other layers
                        op_index=list(op_index[0])
                        if len(op_index)==0:#When no children layers, the layer can be schedule when acc is available
                            time_temp=time_count[acc]
                        else:
                            depend_index=depend_map[op_index,0]
                            time_temp=np.max(np.append(time_count[acc],time_end[bat,depend_index]))
                        time_start[bat,node]=time_temp
                        exe_temp[acc]=exe_time[acc,node]
                        time_end[bat,node]=time_start[bat,node]+exe_temp[acc]
                        time_count[acc]=time_end[bat,node]
                        depend_flag1=np.where(depend_table[bat,:,0]==node) #resolve dependency from current node
                        depend_flag1=list(depend_flag1[0])
                        depend_table[bat,depend_flag1,:]=-1
                        assign_flag[bat,node]=1
                        sche_flag=(np.min(assign_flag)==0)
                        break
        
        time_max=max(time_count)
        time_count[:]=time_max
        
    total_time=np.max(time_end).astype(int)
    
    return time_start,time_end,total_time

def gen_schedule(num_acc,time_start,total_time,best_pos):
    time_start=time_start.astype(int)
    num_batch=time_start.shape[0]
    num_layer=time_start.shape[1]
    schedule=np.ones([num_acc,total_time,2])*(-1)
    for bat in range(num_batch):
        for node in range(num_layer):
            var=node+bat*num_layer
            acc=best_pos[var]
            time_step=time_start[bat,node]
            schedule[acc,time_step,:]=np.array([bat,node])
    return schedule.astype(int)

#The cost function calculates the overall latency and throughput based on the execution time of each layer
def cost_func(schedule,layer_cycle,total_ops,num_batch):
    num_acc=schedule.shape[0]
    total_time=schedule.shape[1]
    time_table=np.ones([num_acc,total_time])*(-1)
    for acc in range(num_acc):
        for step in range(total_time):
            bat,node=schedule[acc,step,:]
            if bat==-1:
                time_table[acc,step]=0
            else:
                time_table[acc,step]=layer_cycle[node]
    
    time_step=np.max(time_table,axis=0)
    time_pipeline=sum(time_step)
    throughput=total_ops*num_batch/(time_pipeline)
    
    return time_table,time_pipeline,throughput


def cal_overhead(acc_pre,acc_nxt,op,hw_config,DATA_TYPE):
    overhead = 0
    freq =230
    if acc_pre!=-1:                                                                                                         
        abc_pre=hw_config[acc_pre,3:6].copy()#a,b,c
        abc_nxt=hw_config[acc_nxt,3:6].copy()#a,b,c                                                               # 0,  1,  2, 3, 4, 5, 6, 7, 8,  9,
        conf_pre=np.concatenate((hw_config[acc_pre,0:6],hw_config[acc_pre,10:13],hw_config[acc_pre,31:32]),axis=0)#h1, w1, w2, a, b, c, X, Y, Z, dup,
        conf_nxt=np.concatenate((hw_config[acc_nxt,0:6],hw_config[acc_nxt,10:13],hw_config[acc_nxt,31:32]),axis=0)
        part_nxt=hw_config[acc_nxt,22:26].copy() #part_aL, part_bL, part_bR, part_cR,
        abc_pre[2]=abc_pre[2]*conf_pre[9]//conf_nxt[9] #once the next layer has dup, this layer should be divided by dup
        if op==0:
            if (abc_nxt[0]*part_nxt[0])%abc_pre[0]!=0 and (abc_nxt[1]*part_nxt[1])%abc_pre[2]!=0: # A%a, B%c
                overhead = np.ceil( (1000/freq) * conf_pre[0] * conf_pre[3] * conf_pre[6] * conf_pre[2] * conf_pre[5] * conf_pre[8] / (128/(DATA_TYPE*8)))
            elif (abc_nxt[0]*part_nxt[0])%abc_pre[0]!=0:
                overhead = np.ceil( (1000/freq) * conf_pre[0] * conf_pre[3] * conf_pre[6] * conf_pre[2] * conf_pre[5] * conf_pre[8] / (128/(DATA_TYPE*8))) // abc_pre[2]
            elif (abc_nxt[1]*part_nxt[1])%abc_pre[2]!=0:
                overhead = np.ceil( (1000/freq) * conf_pre[0] * conf_pre[3] * conf_pre[6] * conf_pre[2] * conf_pre[5] * conf_pre[8] / (128/(DATA_TYPE*8))) // abc_pre[0]
            else:
                overhead=0
        elif op==1:
            if (abc_nxt[1]*part_nxt[2])%abc_pre[0]!=0 and (abc_nxt[2]*part_nxt[3])%abc_pre[2]!=0: # B%a, C%c
                overhead = np.ceil( (1000/freq) * conf_pre[0] * conf_pre[3] * conf_pre[6] * conf_pre[2] * conf_pre[5] * conf_pre[8] / (128/(DATA_TYPE*8)))
            elif (abc_nxt[1]*part_nxt[2])%abc_pre[0]!=0:
                overhead = np.ceil( (1000/freq) * conf_pre[0] * conf_pre[3] * conf_pre[6] * conf_pre[2] * conf_pre[5] * conf_pre[8] / (128/(DATA_TYPE*8))) // abc_pre[2]
            elif (abc_nxt[2]*part_nxt[3])%abc_pre[2]!=0:
                overhead = np.ceil( (1000/freq) * conf_pre[0] * conf_pre[3] * conf_pre[6] * conf_pre[2] * conf_pre[5] * conf_pre[8] / (128/(DATA_TYPE*8))) // abc_pre[0]
            else:
                overhead=0
        else:
            if (abc_nxt[2]*part_nxt[3])%abc_pre[0]!=0 and (abc_nxt[1]*part_nxt[2])%abc_pre[2]!=0: # C%a, B%c
                overhead = np.ceil( (1000/freq) * conf_pre[0] * conf_pre[3] * conf_pre[6] * conf_pre[2] * conf_pre[5] * conf_pre[8] / (128/(DATA_TYPE*8)))
            elif (abc_nxt[2]*part_nxt[3])%abc_pre[0]!=0:
                overhead = np.ceil( (1000/freq) * conf_pre[0] * conf_pre[3] * conf_pre[6] * conf_pre[2] * conf_pre[5] * conf_pre[8] / (128/(DATA_TYPE*8))) // abc_pre[2]
            elif (abc_nxt[1]*part_nxt[2])%abc_pre[2]!=0:
                overhead = np.ceil( (1000/freq) * conf_pre[0] * conf_pre[3] * conf_pre[6] * conf_pre[2] * conf_pre[5] * conf_pre[8] / (128/(DATA_TYPE*8))) // abc_pre[0]
            else:
                overhead=0
                
        # if overhead!=0:
        #     print("Pre Acc is", acc_pre, "Pre Config is", hw_config[acc_pre,3:6], abc_pre, hw_config[acc_pre,22:26])
        #     print("Next Acc is", acc_nxt, "Next Config is", hw_config[acc_nxt,3:6], abc_nxt, part_nxt)
        #     print("The Corresponding operation is:", op , " , Overhead is:", overhead)
    
    return overhead
        
def cal_overhead_fpga(acc_pre,acc_nxt,op,hw_config,DATA_TYPE):
    overhead = 0
    freq =250
    BUFF_WIDTH_B = 64 # Currently hardened as well in cdse_on_chip
    DATA_PCAK_B = BUFF_WIDTH_B//(DATA_TYPE*8)
    BUFF_WIDTH_C = 32 # Currently hardened as well in cdse_on_chip
    if acc_pre!=-1:                                                                                                         
        abc_pre=hw_config[acc_pre,0:3].copy()#a,b,c
        abc_nxt=hw_config[acc_nxt,0:3].copy()                                                    # 0, 1, 2, 3, 4, 5,   6,   
        conf_pre=np.concatenate((abc_pre,hw_config[acc_pre,3:6],hw_config[acc_pre,17:18]),axis=0)# a, b, c, X, Y, Z, dup, 
        conf_nxt=np.concatenate((abc_nxt,hw_config[acc_nxt,3:6],hw_config[acc_nxt,17:18]),axis=0)# a, b, c, X, Y, Z, dup, 
        part_nxt=hw_config[acc_nxt,8:12].copy() #part_aL, part_bL, part_bR, part_cR,

        abc_pre[1]=abc_pre[1]//DATA_PCAK_B
        abc_pre[2]=abc_pre[2]*conf_pre[6]//conf_nxt[6] #once the next layer has dup, this layer should be divided by dup
        abc_nxt[1]=abc_nxt[1]//DATA_PCAK_B
        
        if op==0:
            if (abc_nxt[0]*part_nxt[0])%abc_pre[0]!=0 and (abc_nxt[1]*part_nxt[1])%abc_pre[2]!=0: # A%a, B%c
                overhead = np.ceil( (1000/freq) * conf_pre[0] * conf_pre[3]  * conf_pre[2] * conf_pre[5]  / (BUFF_WIDTH_C/(DATA_TYPE*8)))
            elif (abc_nxt[0]*part_nxt[0])%abc_pre[0]!=0:
                overhead = np.ceil( (1000/freq) * conf_pre[0] * conf_pre[3]  * conf_pre[2] * conf_pre[5]  / (BUFF_WIDTH_C/(DATA_TYPE*8))) // abc_pre[2]
            elif (abc_nxt[1]*part_nxt[1])%abc_pre[2]!=0:
                overhead = np.ceil( (1000/freq) * conf_pre[0] * conf_pre[3]  * conf_pre[2] * conf_pre[5]  / (BUFF_WIDTH_C/(DATA_TYPE*8))) // abc_pre[0]
            else:
                overhead=0
        elif op==1:
            if (abc_nxt[1]*part_nxt[2])%abc_pre[0]!=0 and (abc_nxt[2]*part_nxt[3])%abc_pre[2]!=0: # B%a, C%c
                overhead = np.ceil( (1000/freq) * conf_pre[0] * conf_pre[3] *  conf_pre[2] * conf_pre[5]  / (BUFF_WIDTH_B/(DATA_TYPE*8)))
            elif (abc_nxt[1]*part_nxt[2])%abc_pre[0]!=0:
                overhead = np.ceil( (1000/freq) * conf_pre[0] * conf_pre[3] *  conf_pre[2] * conf_pre[5]  / (BUFF_WIDTH_B/(DATA_TYPE*8))) // abc_pre[2]
            elif (abc_nxt[2]*part_nxt[3])%abc_pre[2]!=0:
                overhead = np.ceil( (1000/freq) * conf_pre[0] * conf_pre[3] *  conf_pre[2] * conf_pre[5]  / (BUFF_WIDTH_B/(DATA_TYPE*8))) // abc_pre[0]
            else:
                overhead=0
        else:
            if (abc_nxt[2]*part_nxt[3])%abc_pre[0]!=0 and (abc_nxt[1]*part_nxt[2])%abc_pre[2]!=0: # C%a, B%c
                overhead = np.ceil( (1000/freq) * conf_pre[0] * conf_pre[3] * conf_pre[2] * conf_pre[5] / (64/(DATA_TYPE*8)))
            elif (abc_nxt[2]*part_nxt[3])%abc_pre[0]!=0:
                overhead = np.ceil( (1000/freq) * conf_pre[0] * conf_pre[3] * conf_pre[2] * conf_pre[5] / (64/(DATA_TYPE*8))) // abc_pre[2]
            elif (abc_nxt[1]*part_nxt[2])%abc_pre[2]!=0:
                overhead = np.ceil( (1000/freq) * conf_pre[0] * conf_pre[3] * conf_pre[2] * conf_pre[5] / (64/(DATA_TYPE*8))) // abc_pre[0]
            else:
                overhead=0
                
        # if overhead!=0:
        #     print("Pre Acc is", acc_pre, "Pre Config is", hw_config[acc_pre,0:3], abc_pre, hw_config[acc_pre,8:12])
        #     print("Next Acc is", acc_nxt, "Next Config is", hw_config[acc_nxt,0:3], abc_nxt, hw_config[acc_nxt,8:12])
        #     print("The Corresponding operation is:", op , " , Overhead is:", overhead)
    
    return overhead

def inter_overhead(MODEL_IN,schedule,mem_move,time_table,hw_config,DATA_TYPE,total_ops,num_batch,board_series):
    num_acc=schedule.shape[0]
    image_size = 224
    DDR_BW = 4
    ddr_time=image_size*image_size*3/1024/1024/1024/DDR_BW*1e9
    total_time=time_table.shape[1]
    time_new=time_table.copy()
    for step in range(total_time):
        move=mem_move[step]
        for i in range(len(move)):
            acc_pre,acc_nxt=move[i][0:2]
            op=move[i][3] #For mem_schedule_new:3, mem_schedule:4
            if board_series=="Versal":
                overhead=cal_overhead(acc_pre,acc_nxt,op,hw_config,DATA_TYPE)
            else:
                overhead=cal_overhead_fpga(acc_pre,acc_nxt,op,hw_config,DATA_TYPE)
            time_new[acc_pre,step]=time_new[acc_pre,step]+overhead
    
    for acc in range(num_acc): 
        for step in range(total_time): 
            if schedule[acc,step,1]==0: #When node is zero add ddr_time
                time_new[acc,step]=time_new[acc,step]+ddr_time
    
    time_step=np.max(time_new,axis=0)
    time_pipeline=sum(time_step)
    throughput=total_ops*num_batch/(time_pipeline)
    
    
    return time_new, time_pipeline, throughput
                
# based on the dependency map and the accelerator assignment, return the number of transactions among Accs
def acc_trans(best_pos,depend_map,num_acc):
    num_depend=depend_map.shape[0]
    acc_trans_table=np.zeros([num_acc,num_acc,3]).astype(int) #Record the number of transcations among Accs
    for i in range(num_depend):
        temp=depend_map[i,:]
        acc_pos=temp[0:2]
        type=temp[2]
        if type<=0:
            type=0
        acc0,acc1=best_pos[acc_pos]
        acc_trans_table[acc0,acc1,type]=acc_trans_table[acc0,acc1,type]+1
    
    return acc_trans_table
    