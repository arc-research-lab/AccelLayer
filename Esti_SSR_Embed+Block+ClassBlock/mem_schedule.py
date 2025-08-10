import numpy as np

def mem_pop(mem_pool,index,acc,mem_num): # index:0-4  --> lhs, rhs, out, pre_lhs, pre_rhs
    if len(mem_pool[acc][index%3])==0:
        mem_num=mem_num+1
        mem_index=mem_num
    else:
        mem_index=mem_pool[acc][index%3][0]
        mem_pool[acc][index%3].remove(mem_index)
    return mem_index,mem_num

def mem_push(mem_pool,index,acc,mem_index):
    mem_pool[acc][index%3].append(mem_index)
    
# Memory Allocation with an explict forwarding and mem release stage
def mem_schedule(MODEL_IN,num_batch,num_layer,depend_map,best_pos,schedule):
    num_acc=schedule.shape[0]
    total_time=schedule.shape[1]

    mem_table=np.ones([num_batch,num_layer,2]).astype(int)*(-1)    #Record the LHS and RHS buffer assignment

    mem_util=[] #Record memory utilized in each time steps [acc][step][LHS,RHS,OUT,LHS_NXT,RHS_NXT]
    for acc in range(num_acc):
        lst_2d = []
        for step in range(total_time): 
            lst_1d = []
            for i in range(5):
                lst_1d.append([])
            lst_2d.append(lst_1d)
        mem_util.append(lst_2d)

    mem_move=[]  #Record memory movement between accs in each step [step][acc_pre,acc_nxt,buf_pre,buf_nxt,op_type]
    for step in range(total_time): 
        mem_move.append([])
    
    mem_num=-1#num_acc*5-1
    mem_pool=[]
    for acc in range(num_acc):
        mem_temp=[]
        for i in range(3):
            mem_temp.append([])
        mem_pool.append(mem_temp)
     
    for step in range(total_time):
        for acc in range(num_acc):
            bat,node=schedule[acc,step,:]
            if bat==-1: #The time step is idle
                continue
            if MODEL_IN[node,4]<2:#weights needed, then reserve mem for weights
                mem_table[bat,node,1]=-2
            for i in range(2): #Check LHS,RHS buffer assignment
                if mem_table[bat,node,i]!=-2:
                    if mem_table[bat,node,i]!=-1: #If already assigned, then allocate assigned mem
                        mem_util[acc][step][i].append(mem_table[bat,node,i])
                    else: # allocate a free mem or need a new one
                        mem_temp,mem_num=mem_pop(mem_pool,i,acc,mem_num)
                        mem_util[acc][step][i].append(mem_temp)
                
            mem_temp,mem_num=mem_pop(mem_pool,2,acc,mem_num) #Assign the output buffer
            mem_util[acc][step][2].append(mem_temp)
            mem_push(mem_pool,2,acc,mem_util[acc][step][2][0])
            if step!=total_time-1:
                op_index=np.where(depend_map[:,0]==node) #Find the one that depends on the current layer
                op_index=list(op_index[0])
                for i in op_index: #Resolve dependency
                    layer_nxt=depend_map[i,1]
                    var=layer_nxt+bat*num_layer
                    acc_nxt = best_pos[var]
                    index=depend_map[i,2] + 3
                    if depend_map[i,2]==2:
                        index = 1 + 3 # If denpend type is 2, then it should be right hand side
                    elif depend_map[i,2]<=-1:
                        index = 0 + 3 # If denpend type is -1, then it should be left hand side
                    mem_temp,mem_num=mem_pop(mem_pool,index,acc_nxt,mem_num)
                    mem_util[acc][step][index].append(mem_temp) 
                    mem_table[bat,layer_nxt,index-3]=mem_temp
                    mem_move_temp=[acc,acc_nxt,mem_util[acc][step][2][0],mem_temp,index-3]
                    mem_move[step].append(mem_move_temp)

        for acc in range(num_acc): #Release the current input buffer
            bat,node=schedule[acc,step,:]
            if bat==-1: #The time step is idle
                continue
            if len(mem_util[acc][step][0])!=0:
                mem_push(mem_pool,0,acc,mem_util[acc][step][0][0])
            if len(mem_util[acc][step][1])!=0:
                mem_push(mem_pool,1,acc,mem_util[acc][step][1][0])
            
    return mem_util,mem_move

# Modified memory allocation policy considering no inter-acc communication overhead which should be guaranteed by lcm
def mem_schedule_new(MODEL_IN,num_batch,num_layer,depend_map,best_pos,schedule):
    num_acc=schedule.shape[0]
    total_time=schedule.shape[1]

    mem_table=np.ones([num_batch,num_layer,2]).astype(int)*(-1)    #Record the LHS and RHS buffer assignment
    mem_dealoc=np.ones([num_batch,num_layer,2]).astype(int)*(1)    #Record the remaining time for buffer deallocation

    mem_num=-1# Initialize buffer index
    mem_util=[] #Record memory utilized in each time steps [acc][step][LHS,RHS,OUT]
    mem_pool=[] #Record avialable memory of each accs [acc][LHS,RHS,OUT]
    mem_move=[]  #Record memory movement between accs in each step [step][acc_pre,acc_nxt,buf_index,op_type]
    
    broadcas_flag=0
    if best_pos[0]==best_pos[1] and best_pos[1]==best_pos[2]:
        broadcas_flag=1
        mem_dealoc=mem_dealoc*3
    
    #############Initialization#############
    for acc in range(num_acc):
        lst_2d = []
        for step in range(total_time): 
            lst_1d = []
            for i in range(3):
                lst_1d.append([])
            lst_2d.append(lst_1d)
        mem_util.append(lst_2d)
    
    for step in range(total_time): 
        mem_move.append([])
    
    for acc in range(num_acc):
        mem_temp=[]
        for i in range(3):
            mem_temp.append([])
        mem_pool.append(mem_temp)
    #############Initialization#############
    
    #############Mem Allocation#############
    for step in range(total_time):
        for acc in range(num_acc):
            bat,node=schedule[acc,step,:]
            layer_type = MODEL_IN[node,4].astype(int)
            if bat==-1: #The time step is idle
                continue
            # Assign mem_util based on mem_table and mem_pool
            for i in range(2): #Assign LHS,RHS buffer
                if layer_type<2 and i==1: #if weights needed, then skip RHS assignment
                    continue
                if mem_table[bat,node,i]!=-1: #If already assigned, then allocate assigned mem
                    mem_util[acc][step][i].append(mem_table[bat,node,i]) 
                    if layer_type<=0 and broadcas_flag:
                        mem_dealoc[bat,node+layer_type:node+layer_type+3,i]=mem_dealoc[bat,node+layer_type:node+layer_type+3,i]-1
                    else:
                        mem_dealoc[bat,node,i]=mem_dealoc[bat,node,i]-1 #when buffer is assigned, update the deallocation table
                else: # allocate a free mem or need a new one
                    mem_temp,mem_num=mem_pop(mem_pool,i,acc,mem_num)
                    mem_util[acc][step][i].append(mem_temp)
                    #For the first QKV in the first block, enable buffer broadcasting
                    if layer_type<=0 and broadcas_flag:
                        mem_table[bat,node+layer_type:node+layer_type+3,i]=mem_temp
                        mem_dealoc[bat,node+layer_type:node+layer_type+3,i]=mem_dealoc[bat,node+layer_type:node+layer_type+3,i]-1
                    else:
                        mem_table[bat,node,i]=mem_temp
                        mem_dealoc[bat,node,i]=mem_dealoc[bat,node,i]-1
                        
                        
            # Update out_buf in mem_util, next in_buf in mem_table, and mem_move 
            if step!=total_time-1:
                op_index=np.where(depend_map[:,0]==node) #Find the one that depends on the current layer
                op_index=list(op_index[0])
                cnt_qkv = 0
                for i in op_index: #Resolve dependency
                    layer_nxt=depend_map[i,1]
                    var=layer_nxt+bat*num_layer
                    acc_nxt = best_pos[var]
                    index=depend_map[i,2]
                    if depend_map[i,2]==2:
                        index = 1 # If depend type is 2, then it should be right hand side
                        mem_temp,mem_num=mem_pop(mem_pool,index,acc_nxt,mem_num)
                    elif depend_map[i,2]<=-1:# If depen type is qkv, then enable broadcast
                        index = 0
                        if cnt_qkv == 0 or broadcas_flag==0:
                            mem_temp,mem_num=mem_pop(mem_pool,index,acc_nxt,mem_num)
                            mem_qkv = mem_temp
                        else:
                            mem_temp = mem_qkv
                        cnt_qkv = cnt_qkv + 1
                    else:
                        mem_temp,mem_num=mem_pop(mem_pool,index,acc_nxt,mem_num)
                    mem_table[bat,layer_nxt,index]=mem_temp
                    mem_util[acc][step][2].append(mem_temp)
                    if MODEL_IN[layer_nxt,4]>0: # IF the layer is not QKV then the output buffer will only be used once
                        mem_dealoc[bat,layer_nxt,index]=1 #when mem_table is updated, assign the deallocation table
                    mem_move_temp=[acc,acc_nxt,mem_temp,index]
                    mem_move[step].append(mem_move_temp)
        
        if step>0:
            for acc in range(num_acc): #Release the current input buffer
                bat,node=schedule[acc,step-1,:]
                if bat==-1: #The time step is idle
                    continue
                if mem_dealoc[bat,node,0]==0:
                    mem_push(mem_pool,0,acc,mem_util[acc][step-1][0][0])
                if mem_dealoc[bat,node,1]==0:
                    mem_push(mem_pool,1,acc,mem_util[acc][step-1][1][0])
        # if step==0:
        #     for bat in range(1):
        #         for layer in range(num_layer):
        #             print("Batch: " + str(bat) + ", Layer: " + str(layer), mem_dealoc[bat][layer],mem_table[bat][layer])
    
    # for step in range(total_time):
    #     for acc in range(num_acc):
    #         print("Acc: " + str(acc) + ", Step: " + str(step), mem_util[acc][step])
    
    return mem_util,mem_move

def mem_cnt(mem_util):
    num_acc=len(mem_util)
    mem_num=np.zeros([num_acc,3]).astype(int)
    list_mem=[]
    for acc in range(num_acc):
        list_1d=[]
        for i in range(3):
            list_1d.append([])
        list_mem.append(list_1d)
    time_total=len(mem_util[0])
    for acc in range(num_acc):
        for step in range(time_total):
            for i in range(3): #LHS,RHS,OUT
                for x in mem_util[acc][step][i]:
                    list_mem[acc][i].append(x)
    
    for acc in range(num_acc):
        for i in range(3):
            mem_num[acc,i]=len(list(set(list_mem[acc][i])))
    
    return list_mem,mem_num