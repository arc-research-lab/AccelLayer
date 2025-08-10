import numpy as np
import math


def buff_count_0(MODEL_IN,BRAM,URAM,PART_AL,PART_BL,PART_BR,PART_CR,PART_AO,PART_CO,PACK_IN,PACK_OUT,h1,w1,w2,a,b,c,x,y,z,y_MHA,z_MHA,left_buf,right_buf,out_buf,RAM_TYPE_A,RAM_TYPE_B,RAM_TYPE_C,AXIS_WIDTH_A,AXIS_WIDTH_B,AXIS_WIDTH_C,harden_aie,DATA_TYPE,force_assign,index_assign=0):
    
    #Num of Elements during PL <-> AIE Transmission
    NUM_PER_PORT_A=AXIS_WIDTH_A//(DATA_TYPE*8)
    NUM_PER_PORT_B=AXIS_WIDTH_B//(DATA_TYPE*8)
    NUM_PER_PORT_C=AXIS_WIDTH_C//(DATA_TYPE*8)
    
    #Data Transmission During One Iteration of Graph
    LEFT_SIZE =math.ceil(h1*w1/NUM_PER_PORT_A)
    RIGHT_SIZE=math.ceil(w1*w2/NUM_PER_PORT_B)
    OUT_SIZE  =math.ceil(h1*w2/NUM_PER_PORT_C)
    
    bank_Lb=(a*math.ceil(b/PACK_IN))*(PART_AL*PART_BL)*math.ceil((AXIS_WIDTH_A)/(72//RAM_TYPE_A))*math.ceil(x*y*LEFT_SIZE*PACK_IN/(PART_AL*PART_BL)/(512*RAM_TYPE_A))*left_buf
    bank_Lu=(a*math.ceil(b/PACK_IN))*(PART_AL*PART_BL)*math.ceil((AXIS_WIDTH_A)/72)*math.ceil(x*y*LEFT_SIZE*PACK_IN/(PART_AL*PART_BL)/4096)*left_buf

    bank_Rb=(c*math.ceil(b/PACK_IN))*(PART_BR*PART_CR)*math.ceil(AXIS_WIDTH_B/(72//RAM_TYPE_B))*math.ceil(y_MHA*z_MHA*RIGHT_SIZE*PACK_IN/(PART_BR*PART_CR)/(512*RAM_TYPE_B))*right_buf
    bank_Ru=(c*math.ceil(b/PACK_IN))*(PART_BR*PART_CR)*math.ceil(AXIS_WIDTH_B/72)*math.ceil(y_MHA*z_MHA*RIGHT_SIZE*PACK_IN/(PART_BR*PART_CR)/4096)*right_buf

    bank_Ob=(a*math.ceil(c/PACK_OUT))*(PART_AO*PART_CO)*math.ceil((AXIS_WIDTH_C)/(72//RAM_TYPE_C))*math.ceil(x*z*OUT_SIZE*PACK_OUT/(PART_AO*PART_CO)/(512*RAM_TYPE_C))*out_buf
    bank_Ou=(a*math.ceil(c/PACK_OUT))*(PART_AO*PART_CO)*math.ceil((AXIS_WIDTH_C)/72)*math.ceil(x*z*OUT_SIZE*PACK_OUT/(PART_AO*PART_CO)/4096)*out_buf
    
    bank_Rb_temp=0
    bank_Ru_temp=0
    if harden_aie==0:
        index_layer=np.where(MODEL_IN[:,4]!=2)
        if len(list(index_layer[0]))!=0:
            MODEL_IN_W=MODEL_IN[list(index_layer[0]),:].copy()
            K=np.ceil(np.divide(MODEL_IN_W[:,1],b))
            N=np.ceil(np.divide(MODEL_IN_W[:,2],c))
            WEIGHTS_SIZE=np.sum(np.ceil(np.divide(np.multiply(K,N),NUM_PER_PORT_B)))*PACK_IN
            num_bram=np.ceil((WEIGHTS_SIZE%4096)/(512*RAM_TYPE_B))
            num_uram=WEIGHTS_SIZE//4096
            bank_Rb_temp=(c*math.ceil(b/PACK_IN))*math.ceil(AXIS_WIDTH_B/(72//RAM_TYPE_B))*num_bram
            bank_Ru_temp=(c*math.ceil(b/PACK_IN))*math.ceil(AXIS_WIDTH_B/72)*num_uram

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
    on_chip_total=on_chip_bram+on_chip_uram*8+(1-on_chip_flag)*1e10
    if force_assign==1:
        buf_index=index_assign
    else:    
        buf_index=np.argmin(on_chip_total)
    bram_use=on_chip_bram[buf_index]
    uram_use=on_chip_uram[buf_index]

    return bram_use,uram_use,bank_Rb_temp,bank_Ru_temp,buf_index


def buff_count_1(bram_weights,uram_weights,PART_AL,PART_BL,PART_BR,PART_CR,PART_AO,PART_CO,PACK_IN,PACK_OUT,h1,w1,w2,a,b,c,x,y,z,left_buf,right_buf,out_buf,RAM_TYPE_A,RAM_TYPE_B,RAM_TYPE_C,AXIS_WIDTH_A,AXIS_WIDTH_B,AXIS_WIDTH_C,harden_aie,DATA_TYPE,force_assign,index_assign=0):
    
    #Num of Elements during PL <-> AIE Transmission
    NUM_PER_PORT_A=AXIS_WIDTH_A//(DATA_TYPE*8)
    NUM_PER_PORT_B=AXIS_WIDTH_B//(DATA_TYPE*8)
    NUM_PER_PORT_C=AXIS_WIDTH_C//(DATA_TYPE*8)
    
    #Data Transmission During One Iteration of Graph
    LEFT_SIZE =math.ceil(h1*w1/NUM_PER_PORT_A)
    RIGHT_SIZE=math.ceil(w1*w2/NUM_PER_PORT_B)
    OUT_SIZE  =math.ceil(h1*w2/NUM_PER_PORT_C)
    
    bank_Lb=(a*math.ceil(b/PACK_IN))*(PART_AL*PART_BL)*math.ceil((AXIS_WIDTH_A)/(72//RAM_TYPE_A))*math.ceil(x*y*LEFT_SIZE*PACK_IN/(PART_AL*PART_BL)/(512*RAM_TYPE_A))*left_buf
    bank_Lu=(a*math.ceil(b/PACK_IN))*(PART_AL*PART_BL)*math.ceil((AXIS_WIDTH_A)/72)*math.ceil(x*y*LEFT_SIZE*PACK_IN/(PART_AL*PART_BL)/4096)*left_buf

    bank_Rb=(c*math.ceil(b/PACK_IN))*(PART_BR*PART_CR)*math.ceil(AXIS_WIDTH_B/(72//RAM_TYPE_B))*math.ceil(y*z*RIGHT_SIZE*PACK_IN/(PART_BR*PART_CR)/(512*RAM_TYPE_B))*right_buf
    bank_Ru=(c*math.ceil(b/PACK_IN))*(PART_BR*PART_CR)*math.ceil(AXIS_WIDTH_B/72)*math.ceil(y*z*RIGHT_SIZE*PACK_IN/(PART_BR*PART_CR)/4096)*right_buf

    bank_Ob=(a*math.ceil(c/PACK_OUT))*(PART_AO*PART_CO)*math.ceil((AXIS_WIDTH_C)/(72//RAM_TYPE_C))*math.ceil(x*z*OUT_SIZE*PACK_OUT/(PART_AO*PART_CO)/(512*RAM_TYPE_C))*out_buf
    bank_Ou=(a*math.ceil(c/PACK_OUT))*(PART_AO*PART_CO)*math.ceil((AXIS_WIDTH_C)/72)*math.ceil(x*z*OUT_SIZE*PACK_OUT/(PART_AO*PART_CO)/4096)*out_buf
    
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

def extra_buff_count(part_array,final_config,RAM_TYPE_A,RAM_TYPE_B,RAM_TYPE_C,PACK_IN,PACK_OUT,AXIS_WIDTH_A,AXIS_WIDTH_B,AXIS_WIDTH_C,DATA_TYPE):
    num_acc=part_array.shape[0]
    buff_use=np.zeros([num_acc,3])#bram,uram,index
    buff_use_extra=np.zeros([num_acc,2])#bram,uram,index
    for i in range(num_acc):
        acc = part_array[i,6]
        bram_weights = final_config[acc,36]
        uram_weights = final_config[acc,37]
        PART_AL,PART_BL,PART_BR,PART_CR,PART_AO,PART_CO=part_array[i,0:6]
        h1,w1,w2,a,b,c= final_config[acc,0:6]
        x,y,z = final_config[acc,10:13]
        left_buf,right_buf,out_buf= final_config[acc,28:31]
        harden_aie = final_config[acc,33]
        buff_use[i,:]=buff_count_1(bram_weights,uram_weights,PART_AL,PART_BL,PART_BR,PART_CR,PART_AO,PART_CO,PACK_IN,PACK_OUT,h1,w1,w2,a,b,c,x,y,z,left_buf,right_buf,out_buf,RAM_TYPE_A,RAM_TYPE_B,RAM_TYPE_C,AXIS_WIDTH_A,AXIS_WIDTH_B,AXIS_WIDTH_C,harden_aie,DATA_TYPE,0,0)
        buff_use_extra[i,0]=buff_use[i,0]-final_config[acc,34]
        buff_use_extra[i,1]=buff_use[i,1]-final_config[acc,35]
    total_extra_ram=np.sum(buff_use_extra,axis=0)
    return buff_use,total_extra_ram

def ext_part_check(MODEL_IN,a,b,c,h1,w1,w2,part_aL,part_bL,part_bR,part_cR,part_aO,part_cO):
    aie_lh=h1//8
    aie_lw=w1//2
    aie_rh=w1//8
    aie_rw=w2//2
    aie_oh=h1//8
    aie_ow=w2//2
    num_layer=MODEL_IN.shape[0]
    part_flag=1
    for i in range(num_layer):
        M,K,N=MODEL_IN[i,0:3]
        X=math.ceil(M/(a*h1))
        Y=math.ceil(K/(b*w1))
        Z=math.ceil(N/(c*w2))
        if ((X%part_aL!=0)and(aie_lh%part_aL!=0)) or ((Y%part_bL!=0)and(aie_lw%part_bL!=0)) or ((Y%part_bR!=0)and(aie_rh%part_bR!=0)) or ((Z%part_cR!=0)and(aie_rw%part_cR!=0)) or ((X%part_aO!=0)and(aie_oh%part_aO!=0)) or ((Z%part_cO!=0)and(aie_ow%part_cO!=0)):
            part_flag=0
            break
    return part_flag