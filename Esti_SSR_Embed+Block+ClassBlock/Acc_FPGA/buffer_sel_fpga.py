import numpy as np
import math


def buff_count_fpga(MODEL_IN,BRAM,URAM,PART_AL,PART_BL,PART_BR,PART_CR,PART_AO,PART_CO,PACK_IN,PACK_OUT,a,b,c,x,y,z,y_MHA,z_MHA,left_buf,right_buf,out_buf,RAM_TYPE_A,RAM_TYPE_B,RAM_TYPE_C,force_assign,index_assign=0):
    BUFF_WIDTH_IN=PACK_IN*8
    BUFF_WIDTH_OUT=PACK_OUT*8
    
    bank_Lb=(a*math.ceil(b/PACK_IN))*(PART_AL*PART_BL)*math.ceil((BUFF_WIDTH_IN)/(72//RAM_TYPE_A))*math.ceil(x*y/(PART_AL*PART_BL)/(512*RAM_TYPE_A))*left_buf
    bank_Lu=(a*math.ceil(b/PACK_IN))*(PART_AL*PART_BL)*math.ceil((BUFF_WIDTH_IN)/72)*math.ceil(x*y/(PART_AL*PART_BL)/4096)*left_buf

    bank_Rb=(c*math.ceil(b/PACK_IN))*(PART_BR*PART_CR)*math.ceil(BUFF_WIDTH_IN/(72//RAM_TYPE_B))*math.ceil(y_MHA*z_MHA/(PART_BR*PART_CR)/(512*RAM_TYPE_B))*right_buf
    bank_Ru=(c*math.ceil(b/PACK_IN))*(PART_BR*PART_CR)*math.ceil(BUFF_WIDTH_IN/72)*math.ceil(y_MHA*z_MHA/(PART_BR*PART_CR)/4096)*right_buf

    bank_Ob=(a*c)*(PART_AO*PART_CO)*math.ceil((BUFF_WIDTH_OUT)/(72//RAM_TYPE_C))*math.ceil(x*z/(PART_AO*PART_CO)/(512*RAM_TYPE_C))*out_buf
    bank_Ou=(a*c)*(PART_AO*PART_CO)*math.ceil((BUFF_WIDTH_OUT)/72)*math.ceil(x*z/(PART_AO*PART_CO)/4096)*out_buf
    
    bank_Rb_temp=0
    bank_Ru_temp=0
    index_layer=np.where(MODEL_IN[:,4]!=2)
    if len(list(index_layer[0]))!=0:
        MODEL_IN_W=MODEL_IN[list(index_layer[0]),:].copy()
        K=np.ceil(np.divide(MODEL_IN_W[:,1],b))
        N=np.ceil(np.divide(MODEL_IN_W[:,2],c))
        WEIGHTS_SIZE=np.sum(np.multiply(K,N))
        num_bram=np.ceil((WEIGHTS_SIZE%4096)/(512*RAM_TYPE_B))
        num_uram=WEIGHTS_SIZE//4096
        bank_Rb_temp=(c*math.ceil(b/PACK_IN))*math.ceil(BUFF_WIDTH_IN/(72//RAM_TYPE_B))*num_bram
        bank_Ru_temp=(c*math.ceil(b/PACK_IN))*math.ceil(BUFF_WIDTH_IN/72)*num_uram

    on_chip_bram= np.zeros([8])
    on_chip_uram= np.zeros([8])
    
    on_chip_bram[0]=math.ceil(bank_Lb+bank_Rb+bank_Ob)# [0,0,0]
    on_chip_uram[0]=0

    on_chip_bram[1]=math.ceil(bank_Lb+bank_Rb) # [0,0,1]
    on_chip_uram[1]=math.ceil(bank_Ou)

    on_chip_bram[2]=math.ceil(bank_Lb+bank_Ob)  # [0,1,0]
    on_chip_uram[2]=math.ceil(bank_Ru)

    on_chip_bram[3]=math.ceil(bank_Lb)    # [0,1,1]
    on_chip_uram[3]=math.ceil(bank_Ru+bank_Ou)

    on_chip_bram[4]=math.ceil(bank_Rb+bank_Ob)  # [1,0,0]
    on_chip_uram[4]=math.ceil(bank_Lu)

    on_chip_bram[5]=math.ceil(bank_Rb)   # [1,0,1]
    on_chip_uram[5]=math.ceil(bank_Lu+bank_Ou)

    on_chip_bram[6]=math.ceil(bank_Ob)     # [1,1,0]
    on_chip_uram[6]=math.ceil(bank_Lu+bank_Ru)

    on_chip_bram[7]=0 # [1,1,1]
    on_chip_uram[7]=math.ceil(bank_Lu+bank_Ru+bank_Ou)
    
    on_chip_bram=on_chip_bram+bank_Rb_temp
    on_chip_uram=on_chip_uram+bank_Ru_temp
    
    on_chip_flag=np.logical_and((on_chip_bram<=BRAM),(on_chip_uram<=URAM))
    on_chip_total=on_chip_bram*8+on_chip_uram+(1-on_chip_flag)*1e10
    
    if force_assign==1:
        buf_index=index_assign
    else:    
        buf_index=np.argmin(on_chip_total)
    bram_use=on_chip_bram[buf_index]
    uram_use=on_chip_uram[buf_index]

    return bram_use,uram_use,bank_Rb_temp,bank_Ru_temp,buf_index


def buff_count_fpga1(bram_weights,uram_weights,PART_AL,PART_BL,PART_BR,PART_CR,PART_AO,PART_CO,PACK_IN,PACK_OUT,a,b,c,x,y,z,left_buf,right_buf,out_buf,RAM_TYPE_A,RAM_TYPE_B,RAM_TYPE_C,force_assign,index_assign=0):
    BUFF_WIDTH_IN=PACK_IN*8
    BUFF_WIDTH_OUT=PACK_OUT*8
    
    bank_Lb=(a*math.ceil(b/PACK_IN))*(PART_AL*PART_BL)*math.ceil((BUFF_WIDTH_IN)/(72//RAM_TYPE_A))*math.ceil(x*y/(PART_AL*PART_BL)/(512*RAM_TYPE_A))*left_buf
    bank_Lu=(a*math.ceil(b/PACK_IN))*(PART_AL*PART_BL)*math.ceil((BUFF_WIDTH_IN)/72)*math.ceil(x*y/(PART_AL*PART_BL)/4096)*left_buf

    bank_Rb=(c*math.ceil(b/PACK_IN))*(PART_BR*PART_CR)*math.ceil(BUFF_WIDTH_IN/(72//RAM_TYPE_B))*math.ceil(y*z/(PART_BR*PART_CR)/(512*RAM_TYPE_B))*right_buf
    bank_Ru=(c*math.ceil(b/PACK_IN))*(PART_BR*PART_CR)*math.ceil(BUFF_WIDTH_IN/72)*math.ceil(y*z/(PART_BR*PART_CR)/4096)*right_buf

    bank_Ob=(a*c)*(PART_AO*PART_CO)*math.ceil((BUFF_WIDTH_OUT)/(72//RAM_TYPE_C))*math.ceil(x*z/(PART_AO*PART_CO)/(512*RAM_TYPE_C))*out_buf
    bank_Ou=(a*c)*(PART_AO*PART_CO)*math.ceil((BUFF_WIDTH_OUT)/72)*math.ceil(x*z/(PART_AO*PART_CO)/4096)*out_buf
    
    on_chip_bram= np.zeros([8])
    on_chip_uram= np.zeros([8])
    
    on_chip_bram[0]=math.ceil(bank_Lb+bank_Rb+bank_Ob)# [0,0,0]
    on_chip_uram[0]=0

    on_chip_bram[1]=math.ceil(bank_Lb+bank_Rb) # [0,0,1]
    on_chip_uram[1]=math.ceil(bank_Ou)

    on_chip_bram[2]=math.ceil(bank_Lb+bank_Ob)  # [0,1,0]
    on_chip_uram[2]=math.ceil(bank_Ru)

    on_chip_bram[3]=math.ceil(bank_Lb)    # [0,1,1]
    on_chip_uram[3]=math.ceil(bank_Ru+bank_Ou)

    on_chip_bram[4]=math.ceil(bank_Rb+bank_Ob)  # [1,0,0]
    on_chip_uram[4]=math.ceil(bank_Lu)

    on_chip_bram[5]=math.ceil(bank_Rb)   # [1,0,1]
    on_chip_uram[5]=math.ceil(bank_Lu+bank_Ou)

    on_chip_bram[6]=math.ceil(bank_Ob)     # [1,1,0]
    on_chip_uram[6]=math.ceil(bank_Lu+bank_Ru)

    on_chip_bram[7]=0 # [1,1,1]
    on_chip_uram[7]=math.ceil(bank_Lu+bank_Ru+bank_Ou)
    
    on_chip_bram=on_chip_bram+bram_weights
    on_chip_uram=on_chip_uram+uram_weights

    on_chip_total=on_chip_bram+on_chip_uram*8
    if force_assign==1:
        buf_index=index_assign
    else:    
        buf_index=np.argmin(on_chip_total)
    bram_use=on_chip_bram[buf_index]
    uram_use=on_chip_uram[buf_index]

    return bram_use,uram_use,buf_index

def extra_buff_fpga(part_array,final_config,RAM_TYPE_A,RAM_TYPE_B,RAM_TYPE_C,DATA_PCAK_B,DATA_PCAK_C):
    num_acc=part_array.shape[0]
    buff_use=np.zeros([num_acc,3])#bram,uram,index
    buff_use_extra=np.zeros([num_acc,2])#bram,uram,index
    for i in range(num_acc):
        acc = part_array[i,6]
        bram_weights = final_config[acc,25]
        uram_weights = final_config[acc,26]
        PART_AL,PART_BL,PART_BR,PART_CR,PART_AO,PART_CO=part_array[i,0:6]
        a,b,c,x,y,z = final_config[acc,0:6]
        left_buf,right_buf,out_buf= final_config[acc,14:17]
        buff_use[i,:]=buff_count_fpga1(bram_weights,uram_weights,PART_AL,PART_BL,PART_BR,PART_CR,PART_AO,PART_CO,DATA_PCAK_B,DATA_PCAK_C,a,b,c,x,y,z,left_buf,right_buf,out_buf,RAM_TYPE_A,RAM_TYPE_B,RAM_TYPE_C,0,0)
        buff_use_extra[i,0]=buff_use[i,0]-final_config[acc,23]
        buff_use_extra[i,1]=buff_use[i,1]-final_config[acc,24]
    total_extra_ram=np.sum(buff_use_extra,axis=0)
    return buff_use,total_extra_ram