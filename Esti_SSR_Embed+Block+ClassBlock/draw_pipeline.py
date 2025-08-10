import numpy as np
from matplotlib import pyplot as plt
import matplotlib 

#Draw Pipeline figure for Each time step
def draw_pipeline(num_acc,time_start,time_total,parent):
    num_batch=time_start.shape[0]
    num_layer=time_start.shape[1]
    parent=np.squeeze(parent)
    fig, ax = plt.subplots()
    groups=[]
    for acc in range(num_acc):
        name= 'Acc' + str(acc)
        groups.append(name)
    
    data=np.ones([num_acc,time_total])
    color=np.ones([time_total,num_acc,3])
    text_val=np.zeros([num_acc,time_total])
    colors=['salmon','lightskyblue','mediumpurple','sandybrown','royalblue','violet','lightseagreen','gold']
    length_color=len(colors)
    for bat in range(num_batch):
        for node in range(num_layer):
            var = int(node+bat*num_layer)
            acc = int(parent[var])
            pos0= int(time_start[bat,node])
            #data[acc,pos0]=1
            index=int((bat%length_color))
            text_val[acc,pos0] = node
            value=matplotlib.colors.to_rgb(colors[index])
            color[pos0,acc,:]=value 
            
    for i in range(data.shape[1]):
        color_sel = color[i,:,:]
        ax.barh(groups, data[:,i], left = np.sum(data[:,:i], axis = 1),color=color_sel, edgecolor = "black", linewidth = 2)
        
    for i, bar in enumerate(ax.patches):
        acc = int(i%num_acc)
        pos0= int(i//num_acc)
        text=text_val[acc,pos0]
        plt.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2 + bar.get_y(),
                int(text), ha = 'center',
                color = 'w', weight = 'bold', size = 10)
        
    fig.show() 
    
def draw_search(bestcost_iter):
    fig, ax = plt.subplots()
    x=np.arange(bestcost_iter.shape[0])
    ax.plot(x,bestcost_iter)
    
    # Provide the title for the semilogy plot
    fig.suptitle('Iteration Vs Time step')
    
    # Give x axis label for the semilogy plot
    ax.set_xlabel('Iteration')
    
    # Give y axis label for the semilogy plot
    ax.set_ylabel('Time Amount')
    
    # Display the semilogy plot
    fig.show()