
# coding: utf-8
#%matplotlib inline
import networkx as nx,matplotlib as mpl,matplotlib.pyplot as plt,random,math,copy
from random import randrange,randint,seed; seed(0)

gna=nx.get_node_attributes; sna=nx.set_node_attributes
size=9; dice_times=20; colors="red green yellow blue purple cyan brown".split(' ')

def find_free(nodeset,x,y):
    global size; cx=x; cy=y
    if (x,y) in nodeset: return (x,y)
    while True:
        cy+=1
        if (cy>=size): 
            cy=0;cx+=1
            if (cx>=size): cx=0
        if cx==x and cy==y:
            return None
        if (cx,cy) in nodeset:
            return (cx,cy)            
def throw_dice(nodeset,n):
    global size
    thrown=dict()
    n=min(len(nodeset), n)
    nodeset_copy=set.copy(nodeset)
    for i in range(n):
        _x= randrange(size); _y=randrange(size)
        free_node=find_free(nodeset_copy,_x,_y)
        assert free_node!=None
        if free_node in thrown:
            display(thrown)
            display((_x,_y))
            display(free_node)
        assert free_node not in thrown
        nodeset_copy.remove(free_node)
        thrown[free_node]=colors[randrange(len(colors))]
    return thrown
G=nx.grid_graph([size,size])
labels = dict(zip(G,[ 10*(int(x[0])+1)+int(x[1])+1 for x in list(G.nodes) ]))
nx.set_node_attributes(G,labels,'label')
pos= dict(zip(G,[ (x*1,y*1) for x,y in list(G) ] ))
nx.set_node_attributes(G,pos,'pos')

thrown=throw_dice(set(G.nodes),dice_times)
SG=G.subgraph(thrown.keys()).copy()
nx.set_node_attributes(SG,thrown,'colors')
G.remove_nodes_from(SG)
nx.set_node_attributes(G,nx.current_flow_betweenness_centrality(G),'bw_c')


sp = nx.shortest_path(G, (1,5), (6,8))



text={ (x,y):(x,y+.3) for x,y in gna(G,'pos') };
plt.figure(1,figsize=(8,8))
nx.draw_networkx(G,pos=gna(G,'pos'), with_labels=False, node_shape='s',cmap=plt.get_cmap('plasma'), node_color=[gna(G,'bw_c').get(x,0)for x in list(G)],node_size=100)
nx.draw_networkx(SG,pos=gna(SG,'pos'), with_labels=False, node_shape='o', node_color=[ mpl.colors.cnames[y] for x,y in gna(SG,'colors').items()],node_size=500)
_=nx.draw_networkx_labels(G, pos=text, labels={ (x,y):str((x,y)) for x,y in gna(G,'pos') })
plt.show()
