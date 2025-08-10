import numpy as np
import math
import sys
sys.path.append('/home/jinming/FPGA24/EA_Search_No_broadcast')
from broadcast_tuning import *
from .buffer_sel_fpga import *


def cdse_on_chip_fpga(MODEL_IN,final_config,HW_Part,final_config_abc,acc_trans_table,index_acc,flag_last_acc,att_flag,Bdot_en,Bdot_num,DATA_TYPE,buff_num,heads,term):

    ################ Hardware Constraints ################
    force_assign=0
    buf_sel=[0]
    DDR_BANK=HW_Part[0]
    DSP_factor=5 if DATA_TYPE==4 else 1
    DSP_NUM=math.floor(HW_Part[1]/DSP_factor)
    BRAM_ALL=math.floor(HW_Part[2])
    URAM_ALL=math.floor(HW_Part[3])
    DBUFF_L=buff_num[0]
    DBUFF_R=1#buff_num[1]
    DBUFF_O=buff_num[2]
    
    #Buffer Configurations
    RAM_TYPE_A=1 #"1":1P "2":T2P
    RAM_TYPE_B=1
    RAM_TYPE_C=2

    # PL Frequency
    freq_rate=250

    ###################### Initialization ###########################
    max_int = 1e30#2147483647
    best_time=max_int #maximum int32 number, maybe need to fix here
    cnt_choice=0
    cnt_best=0
    num_term=24
    sample_num=MODEL_IN.shape[0]
    num_design_best=50    #At most 50 choices reserved for the current best
    num_design_choice=50  #At most 50 choices reserved for the near best point
    temp_cycle=np.zeros([sample_num])
    config=np.ones([num_design_best+num_design_choice,num_term+sample_num])*max_int
    
    #find the pre_acc and op_type that transfer data to acc
    index_trans=np.where(acc_trans_table[:,index_acc,:]!=0)
    acc_trans_pre=np.array(index_trans).transpose()
    index_trans=np.where(acc_trans_table[index_acc,:,:]!=0)
    acc_trans_nxt=np.array(index_trans).transpose()

    ############################ DSE Kernel0 ###############################
    BUFF_WIDTH_B =64 # Currently hardened as well in inter_overhead
    DATA_PCAK_B = BUFF_WIDTH_B//(DATA_TYPE*8)
    BUFF_WIDTH_C =32 # Currently hardened as well in inter_overhead
    DATA_PCAK_C = BUFF_WIDTH_C//(DATA_TYPE*8)
    BUFF_WIDTH_B_MAX=512
    DATA_PCAK_B_MAX=BUFF_WIDTH_B_MAX//(DATA_TYPE*8)
    b_lb=int(math.log2(DATA_PCAK_B))
    b_ub=int(math.log2(DATA_PCAK_B_MAX))
    c_ub=500
    
    Bdot_flag=min(MODEL_IN[:,4])>=2
    
    if Bdot_flag:
        iter_bdot=max(MODEL_IN[:,3]).astype(int)
    else:
        iter_bdot=1
    
    length=acc_trans_nxt.shape[0]
    part_array=np.ones([length,7])
    buff_use=np.ones([length,3])
    part_array_temp=np.ones([num_design_best+num_design_choice,length,7])
    buff_use_temp=np.ones([num_design_best+num_design_choice,length,3])
    BRAM=BRAM_ALL
    URAM=URAM_ALL
    
    new_partition_flag=1
    if new_partition_flag:
        DBUFF_O=0
    
    for c in range(1,(c_ub)+1):
        #Specialization for QKV
        if att_flag ==1:
            if Bdot_en!=0:
                if c%(Bdot_num)!=0:
                    continue
            else:
                if (heads%c!=0) and (c%heads!=0):
                    continue
        for b in (2**p for p in range(b_lb, b_ub+1)):
            for a in range(1,(DSP_NUM//(b*c))+1):
                for dup in range(1,iter_bdot+1):
                    if a*b*c*dup>DSP_NUM:
                        break
                    if a>32 or c>32 or (a%3!=0 and a%2!=0) or (c%3!=0 and c%2!=0):
                        continue 
                    if (heads%dup!=0):
                        continue
                    if new_partition_flag==0:
                        array_flag=aie_dsp_partition(a,b,c,DATA_PCAK_B,final_config_abc,acc_trans_pre,dup,index_acc)
                        if array_flag==0:
                            continue
                    if flag_last_acc:
                        if new_partition_flag==0:
                            array_flag=aie_dsp_partition_check(a,b,c,DATA_PCAK_B,final_config_abc,acc_trans_nxt,dup,index_acc)
                            if array_flag==0:
                                continue 
                            part_array=bram_partition_check(a,b,c,DATA_PCAK_B,final_config_abc,acc_trans_nxt,dup,index_acc) #Fine tune the bank partitioning of the previou layer
                        else:
                            part_array=bram_partition_check_lcm(a,b,c,DATA_PCAK_B,final_config_abc,acc_trans_nxt,dup,index_acc)
                        buff_use,total_extra_ram=extra_buff_fpga(part_array,final_config,RAM_TYPE_A,RAM_TYPE_B,RAM_TYPE_C,DATA_PCAK_B,DATA_PCAK_C) #Update the buffer utilization
                        BRAM=BRAM_ALL-total_extra_ram[0]
                        URAM=URAM_ALL-total_extra_ram[1]
                        
                    if new_partition_flag==0: 
                        part_aL,part_bL,part_bR,part_cR,part_aO,part_cO=bram_partition(a,b,c,DATA_PCAK_B,final_config_abc,acc_trans_pre,dup,index_acc)
                    else:
                        part_aL,part_bL,part_bR,part_cR,part_aO,part_cO=bram_partition_lcm(a,b,c,DATA_PCAK_B,final_config_abc,acc_trans_pre,dup,index_acc)
                    
                    M=max(MODEL_IN[:,0])
                    K=max(MODEL_IN[:,1])
                    N=max(MODEL_IN[:,2])
                    
                    index_layer=np.where(MODEL_IN[:,4]>=2) #This part aims to solve the buffer requirement for 
                    if len(list(index_layer[0]))!=0:
                        MODEL_IN_MHA=MODEL_IN[list(index_layer[0]),:].copy()
                        K_MHA=max(MODEL_IN_MHA[:,1])
                        N_MHA=max(MODEL_IN_MHA[:,2])
                    else:
                        K_MHA=0
                        N_MHA=0
                    
                    x=math.ceil(M/a)
                    y=math.ceil(K/b)
                    z=math.ceil(N/c)
                    
                    y_MHA=math.ceil(K_MHA/(b))
                    z_MHA=math.ceil(N_MHA/(c))
                    
                    bram_use,uram_use,bram_weights,uram_weights,buf_index=buff_count_fpga(MODEL_IN,BRAM,URAM,part_aL, part_bL, part_bR, part_cR, part_aO, part_cO,DATA_PCAK_B,DATA_PCAK_C,a,b,c,x,y,z,y_MHA,z_MHA,DBUFF_L,DBUFF_R,DBUFF_O,RAM_TYPE_A,RAM_TYPE_B,RAM_TYPE_C,force_assign,buf_sel)
                    if (bram_use>BRAM or uram_use>URAM):
                        break
                    
                    Lx=np.ceil(np.divide(MODEL_IN[:,0],a))
                    Ly=np.ceil(np.divide(MODEL_IN[:,1],b))
                    Lz=np.ceil(np.divide(MODEL_IN[:,2],c))

                    temp_cycle = np.multiply(np.multiply(Lx,Ly),Lz) * (1000/250)
                    num_call=np.ceil(np.divide(MODEL_IN[:,3],dup))
                    temp0_cycle=np.multiply(temp_cycle,num_call)
                    total_cycle=np.sum(temp0_cycle)
                    
                    if(total_cycle*0.85<=best_time): # Search design near the best time
                        if total_cycle<best_time:   # If it is the current best
                            best_time=total_cycle
                            index=cnt_best%num_design_best
                            cnt_best=cnt_best+1
                        else:
                            index=num_design_best+(cnt_choice%num_design_choice)
                            cnt_choice=cnt_choice+1
                    
                        config[index,0]=total_cycle
                        config[index,1]=a
                        config[index,2]=b
                        config[index,3]=c
                        config[index,4]=x
                        config[index,5]=y
                        config[index,6]=z
                        config[index,7]=a*b*c*dup
                        config[index,8]=bram_use
                        config[index,9]=uram_use
                        config[index,10]=buf_index
                        config[index,11]=part_aL
                        config[index,12]=part_bL
                        config[index,13]=part_bR
                        config[index,14]=part_cR
                        config[index,15]=part_aO
                        config[index,16]=part_cO
                        config[index,17]=DBUFF_L
                        config[index,18]=DBUFF_R
                        config[index,19]=DBUFF_O
                        config[index,20]=dup
                        config[index,21]=att_flag
                        config[index,22]=bram_weights
                        config[index,23]=uram_weights
                        config[index,num_term:num_term+sample_num]=temp0_cycle[:].copy()
                        part_array_temp[index,:,:]=part_array.copy()
                        buff_use_temp[index,:,:]=buff_use.copy()
    
    index_temp = config[:,0].argsort()
    config = config[index_temp].copy()
    part_array_temp = part_array_temp[index_temp]
    buff_use_temp = buff_use_temp[index_temp]
    best_cycle=config[0,0]
    idnex_perf=np.where(config[:,0]<=np.floor(best_cycle*1.08)) #choose the one with top 92% perf
    idnex_perf=list(idnex_perf[0])
    choose_range=len(idnex_perf)
    
    config_temp=config[0:choose_range,:].copy()
    part_array_temp1 = part_array_temp[0:choose_range].copy()
    buff_use_temp1 = buff_use_temp[0:choose_range].copy()
    ram_sum=np.sum((config_temp[:,8],config_temp[:,9]*8),axis=0)
    index_final = ram_sum.argsort()
    config_temp = config_temp[index_final] #Choose the final configuration from the one with top 92% perf but least RAM usage
    part_array_temp1 = part_array_temp1[index_final]
    buff_use_temp1 = buff_use_temp1[index_final]
    part_array_final=part_array_temp1[0]
    buff_use_final=buff_use_temp1[0]
    
    HW_temp=config_temp[0,:] # 0   1   2  3    4    5      6         7       8         9       10        11       12       13      14        15      16      17     18        19          20         21         22      23    24      25            26
    HW=np.zeros([1,term])    # A,  B,  C, X,   Y,   Z,  PACK_IN, PACK_OUT, part_aL, part_bL, part_bR, part_cR, part_aO, part_cO, DBUFF_L, DBUFF_R, DBUFF_O, dup, att_flag, BUFF_SEL0, BUFF_SEL1, BUFF_SEL2, data_type, BRAM, URAM, bram_weights, uram_weights
    HW[0,0:6]=HW_temp[1:7]
    HW[0,6]=DATA_PCAK_B
    HW[0,7]=DATA_PCAK_C
    HW[0,8:19]=HW_temp[11:22]
    HW[0,22]=DATA_TYPE
    HW[0,23]=HW_temp[8]
    HW[0,24]=HW_temp[9]
    HW[0,25]=HW_temp[22]
    HW[0,26]=HW_temp[23]
    
    
    if HW_temp[10]>8:
        buff_sel_temp=[0,0,0]
    else:
        buff_sel_temp=np.binary_repr(HW_temp[10].astype(int), width=3)
    
    for i in range(3):
        if buff_sel_temp[i]=="0":
            HW[0,i+19]=0
        else:
            HW[0,i+19]=1 
    
    HW_Used=[HW_temp[7],HW_temp[8],HW_temp[9]]#DSP,BRAM_URAM

    return HW, HW_temp[num_term:num_term+sample_num],HW_Used,part_array_final,buff_use_final