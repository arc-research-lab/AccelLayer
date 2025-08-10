import numpy as np
from itertools import combinations
from .cdse_on_chip_fpga import *

def cdac_fpga_top(MODEL_IN,DATA_TYPE,num_acc,inter_assign,acc_trans_table,mem_num,HW_Part,term):
    total_ops = np.sum(np.multiply(np.multiply(np.multiply(MODEL_IN[:,0],MODEL_IN[:,1]),MODEL_IN[:,2]),MODEL_IN[:,3]))*2
    num_layer=MODEL_IN.shape[0]
    index=np.zeros(num_acc).astype(int) 
    
    max_int = 1e30#2147483647
    heads = int(max(MODEL_IN[:,3]))

    unique_cnt=len(np.unique(inter_assign))
    if unique_cnt!=num_acc:
        print('Assignment Not Working' ) 
        final_config=np.zeros((num_acc,term))
        layer_cycle = np.ones((num_layer))*max_int
        return final_config, layer_cycle

    final_config_abc = np.zeros((num_acc,9)).astype(int) #a,b,c,part_aL,part_bL,part_bR,part_cR,dup,att_flag
    final_config_abc[:,3:8]=1
    
    Bdot_flag=np.zeros([num_acc]).astype(int)
    iter_bdot=np.zeros([num_acc]).astype(int)
    index_mha=np.ones(num_acc).astype(int)*(-1)
    list_acc_mha=[]
    cnt_acc_mha=0
    #Check if the batch dot can be paralleled or not
    for i in range(num_acc):
        layer_index=np.where(inter_assign==i)
        layer_index=list(layer_index[0])
        MODEL_PART=MODEL_IN[layer_index,:]
        Bdot_flag[i]=min(MODEL_PART[:,4])>=2
        if Bdot_flag[i]==1: #Multi_head layers are mapped to an Acc alone
            iter_bdot[i]=max(MODEL_PART[:,3]) # Record the number of head
            list_acc_mha.append(i) # Search Multi_head layers first
            index_mha[cnt_acc_mha] = i
            cnt_acc_mha = cnt_acc_mha + 1
    
    list_acc=[]
    cnt_acc=0
    for i in range(inter_assign.shape[0]): #sort the order of the Accs by the order of the gragh
        acc_temp=inter_assign[i]
        if acc_temp not in list_acc:
            list_acc.append(acc_temp)
            index[cnt_acc] =  acc_temp
            cnt_acc=cnt_acc+1  

    Bdot_en=len(np.where(Bdot_flag)[0])
    Bdot_num=np.max(iter_bdot)

    if num_acc==1:
        acc=0                   
        att_flag=1 
        flag_last_acc=1
        final_config=np.zeros((num_acc,term))
        final_config_abc[0,8]=att_flag
        final_config,time_layer,HW_Used,part_array,buff_use=cdse_on_chip_fpga(MODEL_IN,final_config,HW_Part,final_config_abc,acc_trans_table,acc,flag_last_acc,att_flag,Bdot_en,Bdot_num,DATA_TYPE,mem_num[acc],heads,term)
        return final_config, time_layer
    else:
        HW_LEFT=HW_Part[1:4].copy()
        HW_Used=np.zeros([3])
        HW_Cur=np.ones([4])*HW_Part[0]
        HW_temp=np.zeros([3])
        total_ops_nxt=total_ops
        final_config = np.zeros((num_acc,term))#.astype(int)
        layer_cycle = np.zeros((num_layer))
        mem_acc=np.sum(mem_num,axis=1)
        mem_total=np.sum(mem_num)
        dup_hint = int(1)
        #Search Multi_head layers first, to get the hint of dup
        for i,acc in enumerate(index_mha):
            if acc == -1:
                continue
            compensation=np.zeros([3])
            layer_index=np.where(inter_assign==acc)
            layer_index=list(layer_index[0])
            MODEL_PART=MODEL_IN[layer_index,:]
            
            # Here only include multi-head layers
            att_flag=0
            final_config_abc[acc,8]=att_flag
            
            total_ops_cur = np.sum(np.multiply(np.multiply(np.multiply(MODEL_PART[:,0],MODEL_PART[:,1]),MODEL_PART[:,2]),MODEL_PART[:,3]))*2
            comp_ratio=total_ops_cur/total_ops
            mem_ratio=mem_acc[acc]/mem_total
            HW_temp[0]=np.multiply(HW_LEFT[0],comp_ratio)
            HW_temp[1:3]=np.multiply(HW_LEFT[1:3],mem_ratio)
            flag_last_acc = 0
            HW_Cur[1:6]=np.add(HW_temp,compensation)
            
            final_config[acc,:],time_layer,HW_Used,part_array,buff_use=cdse_on_chip_fpga(MODEL_PART,final_config,HW_Cur,final_config_abc,acc_trans_table,acc,flag_last_acc,att_flag,Bdot_en,Bdot_num,DATA_TYPE,mem_num[acc],heads,term)
            
            if final_config[acc,0]==max_int:
                print('No solution during dup initialization, acc:' + str(acc) ) 
                break
            dup = final_config[acc,17]
            #print('Dup hint finish, acc: ' + str(acc) + ' dup is: ' + str(dup)) 
            if dup > dup_hint:
                dup_hint=int(dup)
            
        
        for i,acc in enumerate(index):
            compensation=np.zeros([3])
            layer_index=np.where(inter_assign==acc)
            layer_index=list(layer_index[0])
            MODEL_PART=MODEL_IN[layer_index,:]
            
            # Check if there is QKV kernel
            att_index=np.where(MODEL_PART[:,4]<1) 
            att_index=list(att_index[0])
            att_flag=len(att_index)!=0
            final_config_abc[acc,8]=att_flag
            
            total_ops_cur = np.sum(np.multiply(np.multiply(np.multiply(MODEL_PART[:,0],MODEL_PART[:,1]),MODEL_PART[:,2]),MODEL_PART[:,3]))*2
            comp_ratio=total_ops_cur/total_ops_nxt
            total_ops_nxt=total_ops_nxt-total_ops_cur
            mem_ratio=mem_acc[acc]/mem_total
            mem_total=mem_total-mem_acc[acc]
            HW_temp[0]=np.multiply(HW_LEFT[0],comp_ratio)
            HW_temp[1:3]=np.multiply(HW_LEFT[1:3],mem_ratio)
            if i==num_acc-1:
                compensation=np.zeros([3])
                flag_last_acc=1
            else:
                index_comp=np.where(HW_temp<=50)
                index_comp=list(index_comp[0])
                compensation[index_comp]=50
                flag_last_acc=0
            HW_Cur[1:4]=np.add(HW_temp,compensation)
            #print("Acc ",acc, "Hardware Budget is: ", HW_Cur[1:4])
            final_config[acc,:],time_layer,HW_Used,part_array,buff_use=cdse_on_chip_fpga(MODEL_PART,final_config,HW_Cur,final_config_abc,acc_trans_table,acc,flag_last_acc,att_flag,Bdot_en,dup_hint,DATA_TYPE,mem_num[acc],heads,term)
            if final_config[acc,0]==max_int:
                print('No solution found because of Acc' + str(acc) ) 
                #final_config=np.zeros((num_acc,term))
                layer_cycle = np.ones((num_layer))*max_int
                break
            
            final_config_abc[acc,0:3]=final_config[acc,0:3].copy()
            final_config_abc[acc,3:7]=final_config[acc,8:12].copy()
            final_config_abc[acc,7]=final_config[acc,17].copy()
            layer_cycle[layer_index]=time_layer
            HW_LEFT = np.subtract(HW_LEFT,HW_Used)
            #print("Acc ", acc , ", Config is: ", final_config_abc[acc,0:3], final_config_abc[acc,7], final_config[acc,8:12], final_config[acc,23:25])

            if i==num_acc-1:
                num_modified=part_array.shape[0]
                for i in range(num_modified):
                    acc=int(part_array[i,6])
                    final_config[acc,8:14]=part_array[i,0:6] # Update partition factors
                    final_config[acc,23:25]=buff_use[i,0:2]  # Update RAM Usage
                    buff_index = buff_use[i,2]               # Update RAM Strategy
                    if buff_index>8:
                        buff_sel_temp=[0,0,0]
                    else:
                        buff_sel_temp=np.binary_repr(buff_index.astype(int), width=3)
                    for i in range(3):
                        if buff_sel_temp[i]=="0":
                            final_config[acc,i+19]=0
                        else:
                            final_config[acc,i+19]=1
        #print("Layer Cycle is",layer_cycle)
        return final_config, layer_cycle
            





