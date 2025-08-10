import numpy as np
import math
import sys
sys.path.append('/home/ubuntu/FPGA24/EA_Search_No_broadcast')
from broadcast_tuning import *
from .buffer_sel import *


def cdse_on_chip(MODEL_IN,final_config,HW_Part,final_config_abc,acc_trans_table,index_acc,flag_last_acc,att_flag,Bdot_en,Bdot_num,DATA_TYPE,buff_num,heads,term):

    ################ Hardware Constraints ################
    force_assign=0
    buf_sel=[0]
    DDR_BANK=HW_Part[0]
    AIE_NUM=math.floor(HW_Part[1])
    PLIO_IN=math.floor(HW_Part[2])
    PLIO_OUT=math.floor(HW_Part[3])
    BRAM_ALL=math.floor(HW_Part[4])
    URAM_ALL=math.floor(HW_Part[5])
    
    ################ Hardware Setting ################
    #Single AIE Workload Settings
    if DATA_TYPE==1:
        #PI,PK,PJ=[8,16,4]
        PI,PK,PJ=[16,16,16]
        mac=128
        PACK_IN = 1
        PACK_OUT= 1
        II = 2
        kernel_type=6
    elif DATA_TYPE==2:
        #PI,PK,PJ=[16,2,16]
        PI,PK,PJ=[16,16,16]
        mac=32
        PACK_IN = 1
        PACK_OUT= 1
        II = 2
        kernel_type=4
    elif DATA_TYPE==4:
        #PI,PK,PJ=[8,8,2]
        PI,PK,PJ=[8,8,8]
        mac=8
        PACK_IN = 1
        PACK_OUT= 1
        II = 1
        kernel_type=0
    
    #Buffer Configurations
    RAM_TYPE_A=1 #"1":1P "2":T2P
    RAM_TYPE_B=1
    RAM_TYPE_C=2

    AXIS_WIDTH_A=128     # PLIO Port Width from PL <-> AIE
    AXIS_WIDTH_B=128
    AXIS_WIDTH_C=128
    DATA_PCAK_B=1 #This is used in normal FPGA

    # PL Frequency
    freq_rate=230/250

    #print("\n\n\nCurrent Acc is", index_acc, "number of buffer is ", buff_num, "\n\n\n")
    ###################### Initialization ###########################
    max_int = 1e30#2147483647
    best_time=max_int #maximum int32 number, maybe need to fix here
    cnt_choice=0
    cnt_best=0
    num_term=34
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
    up_bd=24
    lw_bd=1
    
    Harden_flag=max(MODEL_IN[:,4])<=1
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
    
    for c in range(lw_bd, up_bd+1):      ##Row Constaint
        #Specialization for QKV
        if att_flag ==1:
            if Bdot_en!=0:
                if c%(Bdot_num)!=0:
                    continue
            else:
                if (heads%c!=0) and (c%heads!=0):
                    continue
        for b in range(lw_bd, up_bd+1): ##Col Constaint
            for a in range(lw_bd, up_bd+1):
                for dup in range(1,iter_bdot+1):
                    if a*b*c*dup>AIE_NUM or (a>8 and c>8) or (a*c>48):
                        break
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
                        buff_use,total_extra_ram=extra_buff_count(part_array,final_config,RAM_TYPE_A,RAM_TYPE_B,RAM_TYPE_C,PACK_IN,PACK_OUT,AXIS_WIDTH_A,AXIS_WIDTH_B,AXIS_WIDTH_C,DATA_TYPE) #Update the buffer utilization
                        BRAM=BRAM_ALL-total_extra_ram[0]
                        URAM=URAM_ALL-total_extra_ram[1]
                        
                    if new_partition_flag==0: 
                        part_aL,part_bL,part_bR,part_cR,part_aO,part_cO=bram_partition(a,b,c,DATA_PCAK_B,final_config_abc,acc_trans_pre,dup,index_acc)
                    else:
                        part_aL,part_bL,part_bR,part_cR,part_aO,part_cO=bram_partition_lcm(a,b,c,DATA_PCAK_B,final_config_abc,acc_trans_pre,dup,index_acc)
                    part_all = part_aL * part_bL * part_bR * part_cR * part_aO * part_cO
                    if part_aL>8 or part_bL>8 or part_bR>8 or part_cR>8 or part_aO>8 or part_cO>8 or part_all>24:
                        continue
                    if kernel_type%2==1:
                        A_BRO = np.ceil(c,2)
                        height=min(c,8)
                        if  a%2==0:
                            C_BRO=2
                        elif a%3==0:
                            C_BRO=3
                        else:
                            C_BRO=1
                        if (b%PACK_IN!=0) or (c%PACK_OUT!=0):
                            continue
                    else:
                        if c<=8:
                            A_BRO= c
                            C_BRO, height = broadC_factor(a,b,c)
                        else:
                            C_BRO= a
                            A_BRO, height = broadC_factor(c,b,a)
                        if ((b>8 and b%4!=0) or b>24):
                            continue
                    ############ Determine A_BRO and C_BRO ###########

                    ############ Verify Placement ###########
                    length=placement_verify(a,b,c,height)
                    if length > 50:
                        continue
                    
                    for h1 in range(PI,64+1,PI):
                        for w1 in range(32,96+1,PK):
                            for w2 in range(PJ,64+1,PJ):
                                part_flag=ext_part_check(MODEL_IN,a,b,c,h1,w1,w2,part_aL,part_bL,part_bR,part_cR,part_aO,part_cO)
                                if part_flag==0:
                                    continue
                                K1=np.ceil(np.divide(MODEL_IN[:,1],b*w1))
                                N1=np.ceil(np.divide(MODEL_IN[:,2],c*w2))
                                mem_lhs=h1*w1*DATA_TYPE
                                mem_rhs=w1*w2*DATA_TYPE
                                mem_rhs1=np.sum(np.multiply(np.multiply(K1*w1,N1*w2),DATA_TYPE))
                                mem_all=(mem_lhs+mem_rhs)*2
                                mem_all1=mem_lhs*2+mem_rhs1
                                
                                if Harden_flag:
                                    if mem_all1<=30*1024:
                                        plio_in_rhs=0
                                        DBUFF_L=buff_num[0]    #"1":single buffer "2":double buffer
                                        DBUFF_R=0
                                        DBUFF_O=buff_num[2]
                                        harden_aie=1
                                    else:
                                        if mem_all<=30*1024:
                                            plio_in_rhs=c*np.ceil(b/PACK_IN)*np.ceil(a/C_BRO)
                                            DBUFF_L=buff_num[0]    #"1":single buffer "2":double buffer
                                            DBUFF_R=1#buff_num[1] #Weights are all pinned on boards, doesn't need more buffers
                                            DBUFF_O=buff_num[2]
                                            harden_aie=0
                                        else:
                                            break
                                else:
                                    if mem_all<=30*1024:
                                        plio_in_rhs=c*np.ceil(b/PACK_IN)*np.ceil(a/C_BRO)
                                        DBUFF_L=buff_num[0]    #"1":single buffer "2":double buffer
                                        DBUFF_R=1#buff_num[1]
                                        DBUFF_O=buff_num[2]
                                        harden_aie=0
                                    else:
                                        break
                                
                                if new_partition_flag:
                                    DBUFF_O=0

                                ############ Calculate PLIO ###########
                                plio_in_lhs= a*np.ceil(b/PACK_IN)*np.ceil(c/A_BRO)
                                plio_in=(plio_in_lhs + plio_in_rhs)*dup
                                plio_out=a*np.ceil(c/PACK_OUT)*dup
                                if plio_in>PLIO_IN or plio_out>PLIO_OUT:
                                    continue
                                
                                if DATA_TYPE ==1:
                                    if w1<=16:
                                        II=5
                                    elif w1<=32:
                                        II=4
                                    elif w1<=48:
                                        II=3
                                    else:
                                        II=2
                                    
                                COMPUTE_CYCLE=30 + (math.ceil(PI*w1*PJ/mac)+II)*(math.ceil(h1/PI)*math.ceil(w2/PJ)-1) + 80 + b  #initial + loop + acc_stall + stream

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
                                
                                x=math.ceil(M/(a*h1))
                                y=math.ceil(K/(b*w1))
                                z=math.ceil(N/(c*w2))
                                
                                y_MHA=math.ceil(K_MHA/(b*w1))
                                z_MHA=math.ceil(N_MHA/(c*w2))
                                               
                                # if index_acc==0 and h1 ==32 and w1==48 and w2==32:
                                #     print('part_aL, part_bL, part_bR, part_cR, part_aO, part_cO,h1,w1,w2,a,b,c,x,y,z,y_MHA,z_MHA,DBUFF_L,DBUFF_R,DBUFF_O')
                                #     print(part_aL, part_bL, part_bR, part_cR, part_aO, part_cO,h1,w1,w2,a,b,c,x,y,z,y_MHA,z_MHA,DBUFF_L,DBUFF_R,DBUFF_O)
                                #     sys.exit()
                                bram_use,uram_use,bram_weights,uram_weights,buf_index=buff_count_0(MODEL_IN,BRAM,URAM,part_aL, part_bL, part_bR, part_cR, part_aO, part_cO,PACK_IN,PACK_OUT,h1,w1,w2,a,b,c,x,y,z,y_MHA,z_MHA,DBUFF_L,DBUFF_R,DBUFF_O,RAM_TYPE_A,RAM_TYPE_B,RAM_TYPE_C,AXIS_WIDTH_A,AXIS_WIDTH_B,AXIS_WIDTH_C,harden_aie,DATA_TYPE,force_assign,buf_sel)
                                if (bram_use>BRAM or uram_use>URAM):
                                    continue
                                Lx=np.ceil(np.divide(MODEL_IN[:,0],(a*h1)))
                                Ly=np.ceil(np.divide(MODEL_IN[:,1],(b*w1)))
                                Lz=np.ceil(np.divide(MODEL_IN[:,2],(c*w2)))
                                AIE_CYCLE=max([(h1*w1*DATA_TYPE//4),(w1*w2*DATA_TYPE//4),COMPUTE_CYCLE])
                                rd=np.multiply(np.multiply(Lx,Ly),Lz)
                                temp_cycle = (250 + rd * AIE_CYCLE + (rd-1) * 67 + 18 * (b+1) + max([(h1*w1*DATA_TYPE//4)*PACK_IN,(w1*w2*DATA_TYPE//4)*PACK_OUT]))/freq_rate # + M * N * DATA_TYPE//4 #On-chip Overhead #+(h1*w2*DATA_TYPE//4)*PACK_OUT

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
                                    config[index,1]=h1
                                    config[index,2]=w1
                                    config[index,3]=w2
                                    config[index,4]=a
                                    config[index,5]=b
                                    config[index,6]=c
                                    config[index,7]=A_BRO
                                    config[index,8]=C_BRO
                                    config[index,9]=x
                                    config[index,10]=y
                                    config[index,11]=z
                                    config[index,12]=length
                                    config[index,13]=height
                                    config[index,14]=plio_in
                                    config[index,15]=plio_out
                                    config[index,16]=a*b*c*dup
                                    config[index,17]=bram_use
                                    config[index,18]=uram_use
                                    config[index,19]=buf_index
                                    config[index,20]=part_aL
                                    config[index,21]=part_bL
                                    config[index,22]=part_bR
                                    config[index,23]=part_cR
                                    config[index,24]=part_aO
                                    config[index,25]=part_cO
                                    config[index,26]=DBUFF_L
                                    config[index,27]=DBUFF_R
                                    config[index,28]=DBUFF_O
                                    config[index,29]=dup
                                    config[index,30]=att_flag
                                    config[index,31]=harden_aie
                                    config[index,32]=bram_weights
                                    config[index,33]=uram_weights
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
    plio_sum=np.sum((config_temp[:,14],config_temp[:,15]),axis=0)
    ram_sum=np.sum((config_temp[:,17],config_temp[:,18]*8),axis=0)
    index_final = ram_sum.argsort()
    config_temp = config_temp[index_final] #Choose the final configuration from the one with top 92% perf but least RAM usage
    part_array_temp1 = part_array_temp1[index_final]
    buff_use_temp1 = buff_use_temp1[index_final]
    part_array_final=part_array_temp1[0]
    buff_use_final=buff_use_temp1[0]
    
    Versal_HW_temp=config_temp[0,:] # 0      1     2    3    4    5     6      7        8        9     10   11   12       13          14       15    16    17    18        19         20         21        22      23        24      25        26       27     28      29      30    31      32        33       34   35      36             37
    Versal_HW=np.zeros([1,term])    # h1,   w1,   w2,   A,   B,   C,  A_BRO, C_BRO,  PACK_IN, PACK_OUT, X,   Y,   Z,  data_type  kernel_type, layer, col, row, height, BUFF_SEL0, BUFF_SEL1, BUFF_SEL2, part_aL, part_bL, part_bR, part_cR, part_aO, part_cO,DBUFF_L,DBUFF_R,DBUFF_O,dup,atten_flag,harden_aie BRAM,URAM, bram_weights, uram_weights
    Versal_HW[0,0:8]=Versal_HW_temp[1:9]
    Versal_HW[0,8]=PACK_IN
    Versal_HW[0,9]=PACK_OUT
    Versal_HW[0,10:13]=Versal_HW_temp[9:12]
    Versal_HW[0,13]=DATA_TYPE
    Versal_HW[0,14]=kernel_type  

    placement=np.zeros([1,4]) #layer,col,row,height 
    col=(50-Versal_HW_temp[12])//2
    row=0
    placement[0,1:4]=[col,row,Versal_HW_temp[13]] 
    
    Versal_HW[0,15:19]=placement  

    if Versal_HW_temp[19]>8:
        buff_sel_temp=[0,0,0]
    else:
        buff_sel_temp=np.binary_repr(Versal_HW_temp[19].astype(int), width=3)
        
    for i in range(3):
        if buff_sel_temp[i]=="0":
            Versal_HW[0,i+19]=0
        else:
            Versal_HW[0,i+19]=1         
    
    Versal_HW[0,22:34]=Versal_HW_temp[20:32]
    Versal_HW[0,34:36]=Versal_HW_temp[17:19]
    Versal_HW[0,36:38]=Versal_HW_temp[32:34]
    HW_Used=[Versal_HW_temp[16],Versal_HW_temp[14],Versal_HW_temp[15],Versal_HW_temp[17],Versal_HW_temp[18]]#AIE,PLIO_IN,PLIO_OUT,BRAM_URAM
    return Versal_HW,config_temp[0,num_term:num_term+sample_num],HW_Used,part_array_final,buff_use_final