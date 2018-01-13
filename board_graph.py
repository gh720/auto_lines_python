import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import random,math,copy
from random import randrange,randint
from matplotlib.widgets import Button
from collections import deque

# def _init():
#     random.seed(0)
#     size=9
#     dice_times=20

gna=nx.get_node_attributes; sna=nx.set_node_attributes



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


class Board_graph:

    G = None
    metrics=dict()

    def make_graph(board):
        a = board.get_array()
        size = board.get_size()
        G = nx.grid_graph([size,size]) # TODO: decouple
        attrs={}
        for i in range(size):
            for j in range(size):
                data={ 'pos': (i,j), 'occupied': a[i][j] } 
                attrs[(i,j)]=data
        nx.set_node_attributes(G,attrs)
        return G   

    def draw(self,axes):
        G,FG,OG = self.G,self.FG,self.OG # FIND: go about inappropriate calls
        plt.sca(axes)
        plt.cla()

        # OG = G.subgraph([ x for x,data in G.nodes.items() if data['occupied']!=None] ).copy()
        # nx.set_node_attributes(OG,gna(OG,'occupied'),'colors')
        # FG=G.copy(); FG.remove_nodes_from(OG)
        # sna(FG,nx.load_centrality(FG),'lc_metric')
        # ap=nx.articulation_points(FG)
        # APG=FG.subgraph(ap).copy()

        text={ (x,y):(x,y+.3) for x,y in gna(G,'pos') };

        plt.figure(1,figsize=(8,8))
        nx.draw_networkx(FG,pos=gna(G,'pos')
            , with_labels=False, node_shape='s',node_size=100, cmap=plt.get_cmap('plasma')
            , node_color=[gna(FG,'lc_metric').get(x,0)for x in list(FG)] )
        nx.draw_networkx(OG,pos=gna(OG,'pos'), with_labels=False, node_shape='o',node_size=500
            , node_color=[ mpl.colors.cnames[y] for x,y in gna(OG,'color').items()] )
        # nx.draw_networkx_edges(TG, gna(TG,'pos'), sp_edges, width=3.0, edge_color='r', arrow_style='>')
        # nx.draw_networkx_nodes(APG, pos=gna(FG,'pos'), node_shape='d', node_color='r', node_size=300)
        nx.draw_networkx_labels(G, pos=text, labels={ (x,y):str((x,y)) for x,y in gna(G,'pos') })
        plt.show()




    def update_graph(self,board): # TODO: decouple
        a = board.get_array()
        size = board.get_size()
        G = self.G = nx.grid_graph([size,size])
        attrs={}
        # import pdb; pdb.set_trace()
        occupied=dict()
        for i in range(size):
            for j in range(size):
                if a[i][j]!=None:
                    occupied[(i,j)]=1
                data={ 'pos': (i,j), 'color': a[i][j] } 
                attrs[(i,j)]=data
        nx.set_node_attributes(self.G,attrs)

        OG = self.OG = G.subgraph(occupied).copy()
        # nx.set_node_attributes(OG,gna(OG,'color'),'colors')
        
        FG = self.FG=G.copy()
        FG.remove_nodes_from(occupied)

        self.metrics['lc']=nx.load_centrality(FG)
        # ap = list(nx.articulation_points(FG))
        # ap_dict = zip(ap,[1]*len(ap))
        # self.metrics['ap']=ap_dict
        # nx.set_node_attributes(OG,gna(OG,'occupied'),'colors')
        # FG=G.copy(); FG.remove_nodes_from(OG)
        # nx.set_node_attributes(G,nx.load_centrality(G),'lc_metric')
        # APG=FG.subgraph(ap).copy()
        return self.metrics

    def path(self,x,y,x2,y2):
        assert self.G!=None
        path = nx.shortest_path(self.G, (x,y), (x2,y2))
        return path


    def all_paths(self, start, end):
        G=self.G
        parents=[]
        queue = deque(start)
        while queue:
            node=queue.popleft()
            if node==end:
                continue
            nei=set(G[node])
            new = nei-enqueued
            seen = nei & enqueued
            enqueued |= new
            if not new: # dead end
                continue


            for child in new:
                queue.append(child)
                parents[child]=[node]
            for child in seen:
                parents[child].append(node)



    # def bfs(g, start):
    #     queue, enqueued = deque([(None, start)]), set([start])
    #     while queue:
    #         parent, n = queue.popleft()
    #         yield parent, n
    #         new = set(g[n]) - enqueued
    #         enqueued |= new
    #         queue.extend([(n, child) for child in new])




    # def BFS(self,start,end,q):
    #     G= self.G
    #     temp_path = [start]
    #     paths=[]
        
    #     q.enqueue(temp_path)
        
    #     while q.IsEmpty() == False:
    #         tmp_path = q.dequeue()
    #         last_node = tmp_path[len(tmp_path)-1]
    #         print tmp_path
    #         if last_node == end:
    #             paths.append(tmp_path)
    #             # print "VALID_PATH : ",tmp_path
    #         for link_node in G[last_node]:
    #             if link_node not in tmp_path:
    #                 new_path = []
    #                 new_path = tmp_path + [link_node]
    #                 q.enqueue(new_path)
