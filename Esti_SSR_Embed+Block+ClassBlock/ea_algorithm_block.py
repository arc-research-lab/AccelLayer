import numpy as np
from inter_schedule import *
from mem_schedule import *
from Acc_Versal import *
from Acc_FPGA import *

def Roulettewheel(prob):
    index = np.random.rand() * np.sum(prob)
    sum = np.cumsum(prob)
    first_ind=np.where(index<=sum)[0][0]
    return first_ind

def SP_Crossover(x0,x1,num_node):
    nVar= x1.shape[0]
    num_batch=nVar//num_node
    index =  np.random.randint(0,num_node)
    
    y0=x0.copy()
    y1=x1.copy()
    temp0=np.concatenate((x0[0:index],x1[index:num_node]))
    temp1=np.concatenate((x1[0:index],x0[index:num_node]))
    
    for bat in range(num_batch):
        y0[bat*num_node:(bat+1)*num_node]=temp0
        y1[bat*num_node:(bat+1)*num_node]=temp1
    
    return y0,y1

def Uniform_Crossover(x0,x1,num_node):
    nVar= x1.shape[0]
    num_batch=nVar//num_node
    alpha = np.random.randint(0,2,num_node)
    
    y0=x0.copy()
    y1=x1.copy()
    
    temp0=np.multiply(x0[0:num_node],alpha)+np.multiply(x1[0:num_node],(1-alpha))
    temp1=np.multiply(x1[0:num_node],alpha)+np.multiply(x0[0:num_node],(1-alpha))
    
    for bat in range(num_batch):
        y0[bat*num_node:(bat+1)*num_node]=temp0
        y1[bat*num_node:(bat+1)*num_node]=temp1
    
    return y0,y1

def mutate(x,mutate_ratio,num_node,num_acc):
    nVar= x.shape[0]
    num_batch=nVar//num_node
    index =  np.random.randint(0,num_node)
    
    flag=np.random.rand(num_node)<=mutate_ratio
    index = np.argwhere(flag)
    
    temp=x[0:num_node].copy()
    
    for i in index:
        flag1=np.random.rand(1)<=0.5
        if flag1:
            step=1
        else:
            step=-1
        temp[index]=(x[index]+step)%num_acc
    
    
    y=np.zeros([nVar])
    for bat in range(num_batch):
        y[bat*num_node:(bat+1)*num_node]=temp
    
    return y

def evolution_search(MODEL_IN,HW_Part,DATA_TYPE,num_acc,num_batch,num_node,num_block,depend_map,nPop,nChild,nVar,beta,mutate_ratio,nIter,term,board_series):
    num_layer=MODEL_IN.shape[0]
    total_ops = np.sum(np.multiply(np.multiply(np.multiply(MODEL_IN[:,0],MODEL_IN[:,1]),MODEL_IN[:,2]),MODEL_IN[:,3]))*2
    #Initialization
    parents=np.zeros([nPop,nVar]).astype(int)
    parents_cost=np.zeros([nPop])
    best_pos=np.zeros([nVar]).astype(int)
    max_int = 1e30#2147483647
    best_cost=max_int
    final_config = np.ones([num_acc,term])*max_int
    final_time_table = max_int
    final_throughput = max_int
    final_schedule = max_int
    final_mem_move = max_int
    best_cost = max_int
    best_pos = max_int
    best_layer_cycle = max_int
    
    #Initailization
    for i in range(nPop):
        flag=1
        while flag:
            temp=np.random.randint(0,num_acc,num_node)#[0,1,2,3,4,5,6,7]#
            unique_cnt=len(np.unique(temp))
            if unique_cnt==num_acc:
                flag=0
        for bat in range(num_batch*num_block):
            parents[i,bat*num_node:(bat+1)*num_node]=temp
        # for blk in range(0,num_block*num_batch):
        #     index=blk*num_node
        #     parents[i,index:index+num_node]=[0,1,1,2,3,4,5,5]#[0,1,2,3,4,5,6,7]
        time_start,time_end,time_total=inter_schedule(num_acc,num_layer,parents[i,:],depend_map)
        acc_trans_table=acc_trans(parents[i,:],depend_map,num_acc)
        schedule=gen_schedule(num_acc,time_start,time_total,parents[i,:])
        mem_util,mem_move=mem_schedule_new(MODEL_IN,num_batch,num_layer,depend_map,parents[i,:],schedule)
        list_mem,mem_num=mem_cnt(mem_util)
        inter_assign=parents[i,0:num_layer]
        if board_series=='Versal':
            hw_config, layer_cycle=cdac_top(MODEL_IN,DATA_TYPE,num_acc,inter_assign,acc_trans_table,mem_num,HW_Part,term)
        else:
            hw_config, layer_cycle=cdac_fpga_top(MODEL_IN,DATA_TYPE,num_acc,inter_assign,acc_trans_table,mem_num,HW_Part,term)
        if layer_cycle[0] != max_int:
            time_table,time_pipeline,throughput=cost_func(schedule,layer_cycle,total_ops,num_batch)
            time_table, time_pipeline, throughput = inter_overhead(MODEL_IN,schedule,mem_move,time_table,hw_config,DATA_TYPE,total_ops,num_batch,board_series)
        else:
            time_pipeline=max_int
        parents_cost[i]=time_pipeline
        if time_pipeline<best_cost:
            # h1,w1,w2,a,b,c,a_bro,c_bro,pack_in,pack_out,x,y,z, data_type, kernel_type,layer=0,col,row,height,buff0,buff1,bufff2,part_a,part_b,part_c,DBUFF_L,DBUFF_R,DBUFF_O,dup,att_flag
            final_config=hw_config.copy()
            final_time_table=time_table.copy()
            final_throughput=throughput.copy()
            final_schedule=schedule.copy()
            final_mem_move=mem_move.copy()
            best_cost=time_pipeline.copy()
            best_pos=parents[i,:].copy()
            best_layer_cycle=layer_cycle.copy()

    print('Finish Parents Intialization')
    #Best cost of Iterations
    bestcost_iter = np.zeros([nIter])
    bestthp_iter = np.zeros([nIter])

    for iter in range(nIter):
        avg_cost=np.mean(parents_cost)
        if avg_cost!=0:
            parents_cost=np.divide(parents_cost,avg_cost)
        probs= np.exp(np.multiply(parents_cost,-beta))

        # Initialize offsprings population
        children=np.zeros([nChild,nVar]).astype(int)
        children_cost=np.zeros([nChild])

        #Crossover
        for k in range(nChild//2):
            
            p1= parents[Roulettewheel(probs)]
            p2= parents[Roulettewheel(probs)]
            

            children[k*2,:],children[k*2+1,:]=Uniform_Crossover(p1,p2,num_node)

        #Mutation
        for j in range(nChild):

            #Perform mutation
            children[j,:]=mutate(children[j],mutate_ratio,num_node,num_acc)


            #Evaluate
            time_start,time_end,time_total=inter_schedule(num_acc,num_layer,children[j,:],depend_map)
            acc_trans_table=acc_trans(children[j,:],depend_map,num_acc)
            schedule=gen_schedule(num_acc,time_start,time_total,children[j,:])
            mem_util,mem_move=mem_schedule_new(MODEL_IN,num_batch,num_layer,depend_map,children[j,:],schedule)
            list_mem,mem_num=mem_cnt(mem_util)
            inter_assign=children[j,0:num_layer]
            if board_series=='Versal':
                hw_config, layer_cycle=cdac_top(MODEL_IN,DATA_TYPE,num_acc,inter_assign,acc_trans_table,mem_num,HW_Part,term)
            else:
                hw_config, layer_cycle=cdac_fpga_top(MODEL_IN,DATA_TYPE,num_acc,inter_assign,acc_trans_table,mem_num,HW_Part,term)
            if layer_cycle[0] != max_int:
                time_table,time_pipeline,throughput=cost_func(schedule,layer_cycle,total_ops,num_batch)
                time_table, time_pipeline, throughput = inter_overhead(MODEL_IN,schedule,mem_move,time_table,hw_config,DATA_TYPE,total_ops,num_batch,board_series)
            else:
                time_pipeline=max_int
            children_cost[j]=time_pipeline
            if time_pipeline<best_cost:
                final_config=hw_config.copy()
                final_time_table=time_table.copy()
                final_throughput=throughput.copy()
                final_schedule=schedule.copy()
                final_mem_move=mem_move.copy()
                best_cost=time_pipeline.copy()
                best_pos=children[j,:].copy()
                best_layer_cycle=layer_cycle.copy()

        #Update global optimal of the current iteration
        bestcost_iter[iter] = best_cost
        bestthp_iter[iter] = final_throughput

        # Merge Parents and Children and select the better ones
        population=np.concatenate((parents, children), axis=0)
        population_cost=np.concatenate((parents_cost, children_cost), axis=0)
        index = np.argsort(population_cost, axis=0)
        population=population[index]
        population_cost=population_cost[index]
        parents=population[0:nPop]
        parents_cost=population_cost[0:nPop]

        print('Iteration ' + str(iter) + ' : Best Cost = ' + str(bestcost_iter[iter]) + ', Throughput = ' + str(bestthp_iter[iter]) + ' GOPS')

    return final_config,final_time_table,final_throughput,final_schedule,final_mem_move,best_cost,best_pos,bestcost_iter,best_layer_cycle