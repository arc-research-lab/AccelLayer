import numpy as np
import math
from functools import reduce
import sys

def find_factor(num,factor_max):
    for i in range(factor_max,0,-1):
        if num%i==0:
            factor=i
            break
    return factor

def broadC_factor(a,b,c):
    if c>4:
        height=c
        if b>12:
            C_BRO=1
        elif b>5:
            C_BRO=find_factor(a,3)
        elif b>2:
            C_BRO=find_factor(a,4)
        else:
            C_BRO=a
    elif (c==4 or c==3):
        if c==4:
            height=8
        else:
            height=6
        if b>12:
            C_BRO=find_factor(a,2)
        elif b>5:
            C_BRO=find_factor(a,4)   
        elif b>2:
            C_BRO=find_factor(a,8)  
        else:
            C_BRO=a
    elif c==2:
        height=8
        if b>12:
            C_BRO=find_factor(a,4)
        elif b>5:
            C_BRO=find_factor(a,8) 
        elif b>2:
            C_BRO=find_factor(a,16) 
        else:
            C_BRO=a
    else:
        height=8
        if b>12:
            C_BRO=find_factor(a,8)
        elif b>5:
            C_BRO=find_factor(a,16) 
        elif b>2:
            C_BRO=find_factor(a,32) 
        else:
            C_BRO=a
    return C_BRO,height

def placement_verify(a,b,c,height):
    col_full=(a*c)//height
    col_left=(a*c)%height
    length=col_full*b + (col_left!=0)*b
    return length


#Verify if the AIE array configuration follows transcation pattern 
def aie_dsp_partition(a,b,c,DATA_PCAK_B,final_config_abc,acc_trans_pre,dup,index_acc):
    length=acc_trans_pre.shape[0]
    b=b//DATA_PCAK_B
    for i in range(length):
        pre_acc,depend_type=acc_trans_pre[i,:]
        if pre_acc == index_acc: #check if current acc depends on itself 
            if depend_type==0:
                if b%c!=0 and c%b!=0:
                    return 0
            elif depend_type==1:
                if a%b!=0 and b%a!=0:
                    return 0
            else:
                if (c%b!=0 and b%c!=0) or (c%a!=0 and a%c!=0):
                    return 0        
        else:
            a_pre,b_pre,c_pre=final_config_abc[pre_acc,0:3]
            b_pre=b_pre//DATA_PCAK_B
            if a_pre!=0:   
                dup_pre=final_config_abc[pre_acc,7]
                c_pre=c_pre*dup_pre
                if c_pre%dup!=0:
                    return 0
                c_pre=c_pre//dup #Once next layer has dup, then it should be divided
                if depend_type==0:
                    if (a_pre%a!=0 and a%a_pre!=0) or (c_pre%b!=0 and b%c_pre!=0):
                        return 0
                elif depend_type==1:
                    if (a_pre%b!=0 and b%a_pre!=0) or (c_pre%c!=0 and c%c_pre!=0):
                        return 0
                else:
                    if (c_pre%b!=0 and b%c_pre!=0) or (a_pre%c!=0 and c%a_pre!=0):
                        return 0
    return 1

def aie_dsp_partition_check(a,b,c,DATA_PCAK_B,final_config_abc,acc_trans_nxt,dup,index_acc):#Guarantee last acc match the first acc
    length=acc_trans_nxt.shape[0]
    b=b//DATA_PCAK_B
    c=c*dup
    for i in range(length):
        nxt_acc,depend_type=acc_trans_nxt[i,:]
        if nxt_acc != index_acc: #if current acc depends on itself, ignore since already covered
            a_nxt,b_nxt,c_nxt=final_config_abc[nxt_acc,0:3]
            b_nxt=b_nxt//DATA_PCAK_B
            dup_nxt=final_config_abc[nxt_acc,7]
            if c%dup_nxt!=0:
                return 0
            c=c//dup_nxt # if next layer has dup, then this layer should be divided by dup
            if depend_type==0:
                if (a_nxt%a!=0 and a%a_nxt!=0) or (b_nxt%c!=0 and c%b_nxt!=0):#   
                    return 0
            elif depend_type==1:
                if (b_nxt%a!=0 and a%b_nxt!=0) or (c_nxt%c!=0 and c%c_nxt!=0):#
                    return 0
            else:
                if (b_nxt%c!=0 and c%b_nxt!=0) or (c_nxt%a!=0 and a%c_nxt!=0):#
                    return 0
    return 1
    
def bram_partition(a,b,c,DATA_PCAK_B,final_config_abc,acc_trans_pre,dup,index_acc):
    length=acc_trans_pre.shape[0]
    part_aL,part_bL,part_bR,part_cR,part_aO,part_cO=[1,1,1,1,1,1]
    b=b//DATA_PCAK_B
    for i in range(length):
        pre_acc,depend_type=acc_trans_pre[i,:]
        if pre_acc == index_acc: #check if current acc depends on itself     
            if depend_type==0:
                if b%c!=0:
                    part_b_temp = c//b
                    if part_b_temp>part_bL:
                        part_bL=part_b_temp
            elif depend_type==1:
                if b%a!=0:
                    part_b_temp = a//b
                    if part_b_temp>part_bR:
                        part_bR=part_b_temp
            else:
                if b%c!=0:
                    part_b_temp = c//b
                    if part_b_temp>part_bR:
                        part_bR=part_b_temp
                if c%a!=0:
                    part_c_temp = a//c
                    if part_c_temp>part_cR:
                        part_cR=part_c_temp   
        else:
            a_pre,b_pre,c_pre=final_config_abc[pre_acc,0:3]
            b_pre=b_pre//DATA_PCAK_B
            if a_pre!=0:   
                dup_pre=final_config_abc[pre_acc,7]
                c_pre=c_pre*dup_pre
                if c_pre%dup!=0:
                    return [8,8,8,8,8,8]
                c_pre=c_pre//dup #Once next layer has dup, then this layer should be divided by dup
                if depend_type==0:
                    if a%a_pre!=0:
                        part_a_temp = a_pre//a
                        if part_a_temp>part_aL:
                            part_aL=part_a_temp
                    if b%c_pre!=0:
                        part_b_temp = c_pre//b
                        if part_b_temp>part_bL:
                            part_bL=part_b_temp
                elif depend_type==1:
                    if b%a_pre!=0:
                        part_b_temp = a_pre//b
                        if part_b_temp>part_bR:
                            part_bR=part_b_temp
                    if c%c_pre!=0:
                        part_c_temp = c_pre//c
                        if part_c_temp>part_cR:
                            part_cR=part_c_temp
                else:
                    if b%c_pre!=0:
                        part_b_temp = c_pre//b
                        if part_b_temp>part_bR:
                            part_bR=part_b_temp
                    if c%a_pre!=0:
                        part_c_temp = a_pre//c
                        if part_c_temp>part_cR:
                            part_cR=part_c_temp
    return part_aL,part_bL,part_bR,part_cR,part_aO,part_cO

def lcm(numbers):
    return reduce((lambda x, y: int(x * y / math.gcd(x, y))), numbers)

def bram_partition_lcm(a,b,c,DATA_PCAK_B,final_config_abc,acc_trans_pre,dup,index_acc):
    length=acc_trans_pre.shape[0]
    part_aL,part_bL,part_bR,part_cR,part_aO,part_cO=[1,1,1,1,1,1]
    b=b//DATA_PCAK_B
    for i in range(length):
        pre_acc,depend_type=acc_trans_pre[i,:]
        if pre_acc == index_acc: #check if current acc depends on itself     
            if depend_type==0:
                part_b_temp = lcm([b*part_bL,c])
                part_bL=part_b_temp//b
            elif depend_type==1:
                part_b_temp = lcm([b*part_bR,a])
                part_bR=part_b_temp//b
            else:
                part_b_temp = lcm([b*part_bR,a])
                part_bR=part_b_temp//b
                part_c_temp = lcm([c*part_cR,a])
                part_cR=part_c_temp//c
        else:
            a_pre,b_pre,c_pre=final_config_abc[pre_acc,0:3]
            b_pre=b_pre//DATA_PCAK_B
            if a_pre!=0:   
                dup_pre=final_config_abc[pre_acc,7]
                c_pre=c_pre*dup_pre
                if c_pre%dup!=0:
                    return [8,8,8,8,8,8]
                c_pre=c_pre//dup #Once next layer has dup, then this layer should be divided by dup
                if depend_type==0:
                    part_a_temp = lcm([a*part_aL,a_pre])
                    part_aL=part_a_temp//a
                    part_b_temp = lcm([b*part_bL,c_pre])
                    part_bL=part_b_temp//b
                elif depend_type==1:
                    part_b_temp = lcm([b*part_bR,a_pre]) 
                    part_bR=part_b_temp//b
                    part_c_temp = lcm([c*part_cR,c_pre])
                    part_cR=part_c_temp//c
                else:
                    part_b_temp = lcm([b*part_bR,c_pre])
                    part_bR=part_b_temp//b
                    part_c_temp = lcm([c*part_cR,a_pre])
                    part_cR=part_c_temp//c
    return part_aL,part_bL,part_bR,part_cR,part_aO,part_cO
        
def bram_partition_check(a,b,c,DATA_PCAK_B,final_config_abc,acc_trans_nxt,dup,index_acc):
    length=acc_trans_nxt.shape[0]
    part_array=np.ones([length,7]).astype(int)#part_aL_nxt,part_bL_nxt,part_bR_nxt,part_cR_nxt,part_aO_nxt,part_cO_nxt,acc_index
    part_array[:,6]=0
    b=b//DATA_PCAK_B
    c=c*dup
    cnt=0
    for i in range(length):
        nxt_acc,depend_type=acc_trans_nxt[i,:]
        if nxt_acc != index_acc: #if current acc depends on itself, ignore since already covered   
            a_nxt,b_nxt,c_nxt=final_config_abc[nxt_acc,0:3]
            b_nxt=b_nxt//DATA_PCAK_B
            part_array[cnt,0:4]=final_config_abc[nxt_acc,3:7].copy()
            if a_nxt!=0:   
                dup_nxt=final_config_abc[nxt_acc,7]
                c=c//dup_nxt #if next layer has dup, then this layer should be divided by dup
                part_array[cnt,6]=nxt_acc
                if depend_type==0:
                    if (a_nxt*part_array[cnt,0])%a!=0:
                        part_a_temp = a//a_nxt
                        if part_a_temp>part_array[cnt,0]:
                            part_array[cnt,0]=part_a_temp
                    if (b_nxt*part_array[cnt,1])%c!=0:
                        part_b_temp = c//b_nxt
                        if part_b_temp>part_array[cnt,1]:
                            part_array[cnt,1]=part_b_temp
                elif depend_type==1:
                    if (b_nxt*part_array[cnt,2])%a!=0:
                        part_b_temp = a//b_nxt
                        if part_b_temp>part_array[cnt,2]:
                            part_array[cnt,2]=part_b_temp
                    if (c_nxt*part_array[cnt,3])%c!=0:
                        part_c_temp = c//c_nxt
                        if part_c_temp>part_array[cnt,3]:
                            part_array[cnt,3]=part_c_temp
                else:
                    if (b_nxt*part_array[cnt,2])%c!=0:
                        part_b_temp = c//b_nxt
                        if part_b_temp>part_array[cnt,2]:
                            part_array[cnt,2]=part_b_temp
                    if (c_nxt*part_array[cnt,3])%a!=0:
                        part_c_temp = a//c_nxt
                        if part_c_temp>part_array[cnt,3]:
                            part_array[cnt,3]=part_c_temp
            cnt=cnt+1
            
    return part_array  

def bram_partition_check_lcm(a,b,c,DATA_PCAK_B,final_config_abc,acc_trans_nxt,dup,index_acc):
    length=acc_trans_nxt.shape[0]
    part_array=np.ones([length,7]).astype(int)#part_aL_nxt,part_bL_nxt,part_bR_nxt,part_cR_nxt,part_aO_nxt,part_cO_nxt,acc_index
    part_array[:,6]=0
    b=b//DATA_PCAK_B
    c=c*dup
    cnt=0
    for i in range(length):
        nxt_acc,depend_type=acc_trans_nxt[i,:]
        if nxt_acc != index_acc: #if current acc depends on itself, ignore since already covered   
            a_nxt,b_nxt,c_nxt=final_config_abc[nxt_acc,0:3]
            b_nxt=b_nxt//DATA_PCAK_B
            part_array[cnt,0:4]=final_config_abc[nxt_acc,3:7].copy()
            if a_nxt!=0:   
                dup_nxt=final_config_abc[nxt_acc,7]
                c=c//dup_nxt #if next layer has dup, then this layer should be divided by dup
                part_array[cnt,6]=nxt_acc
                if depend_type==0:
                    part_a_temp = lcm([a_nxt*part_array[cnt,0],a])
                    part_array[cnt,0]=part_a_temp//a_nxt
                    part_b_temp = lcm([b_nxt*part_array[cnt,1],c])
                    part_array[cnt,1]=part_b_temp//b_nxt
                elif depend_type==1:
                    part_b_temp = lcm([b_nxt*part_array[cnt,2],a])
                    part_array[cnt,2]=part_b_temp//b_nxt
                    part_c_temp = lcm([c_nxt*part_array[cnt,3],c])
                    part_array[cnt,3]=part_c_temp//c_nxt
                else:
                    part_b_temp = lcm([b_nxt*part_array[cnt,2],c])
                    part_array[cnt,2]=part_b_temp//b_nxt
                    part_c_temp = lcm([c_nxt*part_array[cnt,3],a])
                    part_array[cnt,3]=part_c_temp//c_nxt
            cnt=cnt+1
    
    return part_array   