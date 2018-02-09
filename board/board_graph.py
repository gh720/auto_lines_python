import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import random,math,copy
from random import randrange,randint
from matplotlib.widgets import Button
from collections import deque,defaultdict
from attrdict import AttrDict
from typing import List, Set, Dict, Tuple, Text, Optional, AnyStr


from pprint import pprint
from .connect_assess import connect_assessor_c

from .utils import *

# def _init():
#     random.seed(0)
#     size=9
#     dice_times=20

gna=nx.get_node_attributes; sna=nx.set_node_attributes

class Board_graph:

    axes=None
    G = None
    metrics=dict()

    def __init__(self, axes: object = None) -> None:
        self.axes=axes
        self.lock=False


    # def make_graph(board):
    #     a = board.get_array()
    #     size = board.get_size()
    #     G = nx.grid_graph([size,size]) # TODO: decouple
    #     attrs={}
    #     for i in range(size):
    #         for j in range(size):
    #             data={ 'pos': (i,j), 'occupied': a[i][j] } 
    #             attrs[(i,j)]=data
    #     nx.set_node_attributes(G,attrs)
    #     return G   

    def init_drawing(self,axes):
        self.axes=axes
        plt.sca(axes)
        plt.cla()

    # def show(self,axes=None,block=None):
    #     plt.sca(axes or self.axes)
    #     plt.show(block=block)


    def update_graph(self,board): # TODO: decouple
        a = board.get_array()

        # self.scrub_edges=[]
        # for i,tup in enumerate(board._scrubs):
        #     if i>0:
        #         self.scrub_edges.append((board._scrubs[i-1], tup))
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

        # self.metrics['lc']=dict()
        self.metrics['lc']=nx.load_centrality(FG)

        # ap = list(nx.articulation_points(FG))
        # ap_dict = zip(ap,[1]*len(ap))
        # self.metrics['ap']=ap_dict
        # nx.set_node_attributes(OG,gna(OG,'occupied'),'colors')
        # FG=G.copy(); FG.remove_nodes_from(OG)
        # nx.set_node_attributes(G,nx.load_centrality(G),'lc_metric')
        # APG=FG.subgraph(ap).copy()
        return self.metrics

    def change_free_graph(self, board, free_cells=[], fill_cells=[]):
        for pos in free_cells:
            node= (pos.x, pos.y)
            node_data = self.G.nodes[node] # TODO: check intersections later
            assert node not in self.FG.nodes
            self.FG.add_node(node, **node_data)
        for pos in free_cells:
            node = (pos.x, pos.y)
            for nei_node in self.G[node]:
                if nei_node in self.FG:
                    self.FG.add_edge(node,nei_node)
        for pos in fill_cells:
            node = (pos.x, pos.y)
            assert node in self.FG.nodes
            self.FG.remove_node(node)

    def metric(self, metric, node):
        value=None
        if metric=='lc':
            value=nx.load_centrality(self.FG, node)
        return value

    def draw(self,axes=True, chained=False):
        G,FG,OG = self.G,self.FG,self.OG # FIND: go about inappropriate calls
        # if not chained:
        #     plt.sca(axes)
        #     plt.cla()
        if axes==True:
            plt.sca(self.axes)
        elif axes:
            plt.sca(axes)


        # OG = G.subgraph([ x for x,data in G.nodes.items() if data['occupied']!=None] ).copy()
        # nx.set_node_attributes(OG,gna(OG,'occupied'),'colors')
        # FG=G.copy(); FG.remove_nodes_from(OG)
        # sna(FG,nx.load_centrality(FG),'lc_metric')
        # ap=nx.articulation_points(FG)
        # APG=FG.subgraph(ap).copy()

        # text={ (x,y):(x,y+.3) for x,y in gna(G,'pos') };

        
        nx.draw_networkx(FG,pos=gna(G,'pos')
            , with_labels=False, node_shape='s',node_size=100, cmap=plt.get_cmap('plasma')
            , node_color=[gna(FG,'lc_metric').get(x,0)for x in list(FG)] )
        nx.draw_networkx(OG,pos=gna(OG,'pos'), with_labels=False, node_shape='o',node_size=500
            , node_color=[ mpl.colors.cnames[y] for x,y in gna(OG,'color').items()] )
        # nx.draw_networkx_edges(TG, gna(TG,'pos'), sp_edges, width=3.0, edge_color='r', arrow_style='>')
        # nx.draw_networkx_nodes(APG, pos=gna(FG,'pos'), node_shape='d', node_color='r', node_size=300)
        # nx.draw_networkx_labels(G, pos=text, labels={ (x,y):str((x,y)) for x,y in gna(G,'pos') })
        # if not chained:
        #     plt.show()


    def draw_moves(self, board, moves):
        FG=None
        self.draw(axes=False, chained=True)

        for move in moves:
            f=move.cell_from
            t=move.cell_to
            try:
                FG = self.FG.copy()
                start_node=f.x,f.y
                end_node=t.x,t.y
                for node in [start_node,end_node]:
                    node_data = self.G.nodes[node]
                    color = node_data['color']
                    FG.add_node(node, **node_data)
                    for nei_node in self.G[node]:
                        if nei_node in self.FG:
                            FG.add_edge(node, nei_node)
                path=nx.shortest_path(FG, tuple(f), tuple(t))
            except nx.NetworkXNoPath:
                return False
            edges=[]
            for i,node in enumerate(path):
                if i<len(path)-1:
                    edges.append((node,path[i+1]))

            TG = self.G.edge_subgraph(edges)
            # plt.sca(self.axes)
            nx.draw_networkx_edges(TG, gna(self.G, 'pos'), width=2, edge_color=color or 'cyan')
            # if tentative_scrub_edges:
            #     SG = self.G.edge_subgraph(tentative_scrub_edges)
        # g_path = nx.DiGraph()
        # g_path.add_nodes_from(TG)
        #
        # g_path_traceback = g_path.copy()
        #
        # g_path.add_edges_from(edges)
        # g_path_traceback.add_edges_from(edges_traceback)

        # if not chained:
        #     plt.sca(axes)
        #     plt.cla()
        # if tentative_scrub_edges:
        #     nx.draw_networkx_edges(SG, gna(self.G, 'pos'), width=3, style='dotted', edge_color='red')
        # nx.draw_networkx_edges(g_path_traceback, gna(self.G, 'pos'), width=1, edge_color='red')
        # if not chained:
        #     plt.show()
        return True


    def draw_move(self, board, f:ddot, t:dpos, tentative_scrub_edges, chained=False):
        FG=None
        if not f:
            return False
        try:
            FG = self.FG.copy()
            start_node = f.x, f.y
            end_node = t.x, t.y
            for node in [start_node, end_node]:
                node_data = self.G.nodes[node]
                color = node_data['color']
                FG.add_node(node, **node_data)
                for nei_node in self.G[node]:
                    if nei_node in self.FG:
                        FG.add_edge(node, nei_node)
            path=nx.shortest_path(FG, tuple(f), tuple(t))
        except nx.NetworkXNoPath:
            return False
        # parents = self.all_paths(start, end)
        # edges = set()
        # for child, child_parents in parents.items():
        #     for parent in child_parents:
        #         edges.add((parent, child))

        # edges_traceback = [(x, y) for x, y in self.get_paths_edges(parents, end) if x != None]
        edges=[]
        for i,node in enumerate(path):
            if i<len(path)-1:
                edges.append((node,path[i+1]))

        TG = self.G.edge_subgraph(edges)
        if tentative_scrub_edges:
            SG = self.G.edge_subgraph(tentative_scrub_edges)
        # g_path = nx.DiGraph()
        # g_path.add_nodes_from(TG)
        #
        # g_path_traceback = g_path.copy()
        #
        # g_path.add_edges_from(edges)
        # g_path_traceback.add_edges_from(edges_traceback)

        # if not chained:
        #     plt.sca(axes)
        #     plt.cla()
        self.draw(chained=True)
        plt.sca(self.axes)
        nx.draw_networkx_edges(TG, gna(self.G, 'pos'), width=2, edge_color='cyan')
        if tentative_scrub_edges:
            nx.draw_networkx_edges(SG, gna(self.G, 'pos'), width=3, style='dotted', edge_color='red')
        # nx.draw_networkx_edges(g_path_traceback, gna(self.G, 'pos'), width=1, edge_color='red')
        # if not chained:
        #     plt.show()
        return True

    def draw_paths_(self,axes,chained=False):
        start = (0,0)
        end = (5,8)
        parents = self.all_paths(start,end)
        edges = set()
        for child, child_parents  in parents.items():
            for parent in child_parents:
                edges.add((parent,child))

        edges_traceback=[ (x,y) for x,y in self.get_paths_edges(parents,end) if x!=None ]
        TG =self.G.subgraph(list(parents.keys())+[end])
        g_path = nx.DiGraph()
        g_path.add_nodes_from(TG)

        g_path_traceback = g_path.copy()
        
        g_path.add_edges_from(edges)
        g_path_traceback.add_edges_from(edges_traceback)

        # if not chained:
        #     plt.sca(axes)
        #     plt.cla()
        #

        self.draw(axes,chained=True)
        nx.draw_networkx_edges(g_path, gna(self.G,'pos'), width=2, edge_color='cyan')
        nx.draw_networkx_edges(g_path_traceback, gna(self.G,'pos'), width=1, edge_color='red')
        # if not chained:
        #     plt.show()

    def assess_paths(self,start,end):
        FG=self.FG
        self.cycles=nx.cycle_basis(FG)
        self.art = nx.articulation_points(FG)
        short = nx.shortest_path(FG,start,end)
        cutsets = self.get_path_cutsets(short,self.cycles)
        prob = self.cutoff_probability(cutsets)
        
    
    # compute triangle area (x 2), > 0 - right turn, < 0 - left turn
    @staticmethod
    def triangle_area2(a,b,c):
        a.x*b.y -a.y*b.x+a.y*c.x-a.x*c.y+b.x*c.y-c.x*b.y

    @staticmethod
    def get_cycle_bounds(cycle):
        i_min=0
        minx,miny=cycle[0]
        maxx,maxy=cycle[0]
        miny_for_minx=None
        for i in range(len(cycle)):
            x,y = cycle[i]
            if x< minx:
                minx=x
                miny_for_minx=y
                i_min=i
            elif x==minx:
                if miny_for_minx==None or y<miny_for_minx:
                    miny_for_minx=y
                    i_min=i
            miny=min(miny,y)
            maxx=max(maxx,x)
            maxy=max(maxy,y)
        return i_min,minx,miny,maxx,maxy

    # -1 - anti, 1 - clock-wise, None - not a cycle 
    @staticmethod
    def get_cycle_direction(cycle):
        assert len(cycle)>2
        e=AttrDict()
        v0=AttrDict()
        v=AttrDict()
        
        i_min,minx,miny,maxx,maxy=__class__.get_cycle_bounds(cycle)
        i_min_next=(i_min+1)%len(cycle)

        x,y = cycle[i_min]
        e.x0=x
        e.y0=y
        e.x0,e.y0=cycle[i_min]
        e.x1,e.y1=cycle[i_min_next]

        # we've got left bottom corner, the only directions is up (clockwise) and right (anti-clockwise)

        v0.dx,v0.dy = e.x1-e.x0, e.y1-e.y0

        if v0.dy==1:
            return 1
        if v0.dx==1:
            return -1
        assert False
        return None

    @staticmethod
    def change_cycles_direction(cycles, dir):
        new_cycles=[]
        for cycle in cycles:
            old_dir = __class__.get_cycle_direction(cycle)
            new_cycle=copy.copy(cycle)
            if old_dir!=dir:
                new_cycle.reverse()
            new_cycles.append(new_cycle)
        return new_cycles


    @staticmethod
    def get_cycle_sets(cycles):
        sets=AttrDict()
        sets.node_map=defaultdict(dict)
        sets.cycles=[]

        for ci, cycle in  enumerate(cycles):
            d = dict()
            for ni,node in enumerate(cycle):
                sets.node_map[node][ci]=ni
                d[node]=ni
            sets.cycles.append([cycle,d])
        return sets

    @staticmethod
    def segment_key(item):
        pass


    @staticmethod
    def get_path_segments(path,cycle_sets):
        segments=dict()
        for cy_i, item in enumerate(cycle_sets.cycles):
            cycle,cycle_dict=item
            segment= [ [i,node] for i,node in enumerate(path) if node in item[1] ]
            if segment:
                segments[cy_i].append([segment, cy_i])

        return sorted(segments, key=lambda x: x[0][0])

    @staticmethod
    def next_i(_list,list_i):
        if not _list:
            return None
        return (list_i+1)% len(_list)

    @staticmethod
    def next_n(_list,list_i,cycle=True):
        if not _list:
            return None
        i= (list_i+1)% len(_list)
        if i<=list_i and not cycle:
            return None,None
        return _list[i],i

    @staticmethod
    def prev_i(_list,list_i):
        if not _list:
            return None
        return (list_i-1+len(_list))% len(_list)

    @staticmethod
    def prev_n(_list,list_i,cycle=True):
        if not _list:
            return None
        i= (list_i-1+len(_list))% len(_list)
        if i >= list_i and not cycle:
            return None,None
        return _list[i],i

    # @staticmethod
    # def m1(path,segments):
    #     alt_path_segments=[]
    #     for seg_i, item in enumerate(segments):
    #         segment,cycle,cycle_dict=item
    #         path_i,node= segment[0]
    #         next_path_i=next_i(path,path_i)
    #         a,b = [ cycle_dict[path[i]] for i in [path_i,next_path_i]]
    #         if b==next_i(cycle, a):
    #             pass
    #         else:
    #             alt_segment=[]
    #             while True:
    #                 alt_segment.append(a)
    #                 a=next_i(cycle,a)
    #                 if a==b:
    #                     break
    #                 assert len(alt_path)<100

    @staticmethod
    def get_node_cycles(node,cycle_sets):
        pass


    # @staticmethod
    # def check_direction(cycle_sets,cy_i,node,next_node):
    #     nm=cycle_sets.node_map
    #     cn_i=nm[node][cy_i] if node in nm else None
    #     cn_i_next=nm[next_node][cy_i] if next_node in nm else None
    #     if next_i(cycle_sets.cycles[cy_i],cn_i)==cn_i_next:
    #         return 1
    #     if prev_i(cycle_sets.cycles[cy_i],cn_i)==cn_i_next:
    #         return -1
    #     return None


    @staticmethod
    def get_segment(cycle_sets, cy_i, start, dir):
        pass

    @staticmethod
    def get_dag(G,start):
        parents=dict()
        order=dict()
        queue=deque([(None,start)])
        edges=[]
        counter=0
        while queue:
            parent,node =  queue.popleft()
            if node not in order:
                order[node] = counter
                counter+=1
            if node not in parents:
                for n in G[node]:
                    if n!=parent:
                        queue.append((node,n))
            if parent!=None:
                if order[node] < order[parent]:
                    parents.setdefault(parent,[]).append(node)
                    edges.append((node,parent))
                else:
                    parents.setdefault(node,[]).append(parent)
                    edges.append((parent,node))
            
            
        return edges

    @staticmethod
    def minleaf(G,start):
        dag= __class__.get_dag(G,start)

        GG = nx.Graph()
        GG.add_nodes_from(G)
        GG.add_edges_from(dag)
        B=nx.DiGraph()
        tgt_dict =dict()
        mm_tgt_set = set()
        tmin = []
        for edge in dag:
            src= edge[0]
            tgt=(edge[1][0]+100,edge[1][1]+100)
            tgt_dict.setdefault(tgt, []).append(src)
            B.add_edge(src,tgt)
        mm = nx.maximal_matching(B)
        for edge in mm:
            mm_tgt_set.add(edge[1])
        for node in tgt_dict:
            if node not in mm_tgt_set:
                mm.add((tgt_dict[node][0], node))
        for edge in mm:
            tgt=(edge[1][0]-100,edge[1][1]-100)
            assert tgt[0]>=0 and tgt[1]>=0
            tmin.append((edge[0],tgt))

        return tmin,dag


    # def get_cutsets(self,path,cycles,level=0):
    #     if level>2:
    #         return []
    #     FG=self.FG
    #     scheduled=dict()
    #     cycle_sets=__class__.get_cycle_sets(cycles)
    #     get_path_segments(path,cycle_sets)
    #     nm = cycle_sets.node_map
    #     for path_i,node in enumerate(path):
    #         cy_indexes = nm[node]
    #         next_node, next_i = next_n(path,path_i)
    #         if next_node==None:
    #             break
    #         for cy_i in cy_indexes:
    #             if cy_i in scheduled:
    #                 continue
    #             scheduled[cy_i]=1 # need to skip this sometimes
    #             dir = check_direction(cycle_sets,cy_i,node,next_node)
    #             end_node = None
    #             segment = __class__.get_segment(cycle_sets, cy_i, start=node, dir=-dir)
    #             get_cutsets_rv(self,segment)
    #
    #
    #
    #             cycle,node_dict= cycle_sets.cycles[cy_i]
    #
    #             check_direction(node,next_node,cycle)
    #




    def get_cycle_patches(self,path,cycles):
        return None

    # def get_path_cutsets(self, path, cycles, level=0):
    #     if level >2:
    #         return []
    #     FG=self.FG
    #     patches=self.get_cycle_patches(path,cycles)
    #     cutsets=dict()
    #     for i in range(len(patches)):
    #         patch = patches[i]
    #         next_patch=patches[i+1] if i < len(patches)-1 else None
    #         if patch.conn==1:
    #             for node in patch.nodes:
    #                 self.add_cutsets(cutsets,[node])
    #             continue
    #         if patch.conn==2:
    #             if next_patch!=None:
    #                 shared=self.intersect(cycles[patch.cycle],cycles[next_patch.cycle])
    #                 pfw = self.get_cycle_patch(cycles[patch.cycle],start=patch.nodes[0],end=shared[0])
    #                 pback =self.get_cycle_patch(cycles[patch.cycle],start=shared[-1],end=patch.nodes[0])
    #                 trimmed_cycles=[cycles[j] for j in range(len(cycles)) if j!=i]
    #                 fw_cutsets=self.get_path_cutsets(pfw,trimmed_cycles, level+1)
    #                 back_cutsets=self.get_path_cutsets(pback,trimmed_cycles, level+1)
    #                 for i in fw_cutsets:
    #                     for j in back_cutsets:
    #                         self.add_cutsets(cutsets, i+j)


    # def cutoff_probability(self, cutsets):
    #     prob=0
    #     c1 = [ x for x in cutsets if len(x) ==1]
    #     c2 = [ x for x in cutsets if len(x) ==2]
    #     c3 = [ x for x in cutsets if len(x) ==3]
    #     node_count=len(self.FG.nodes)
    #     prob = c1/node_count
    #     if node_count>1:
    #         prob += c2 / (node_count*(node_count-1))
    #
    #
    #     if node_count>2:
    #         prob += c3 / (node_count*(node_count-1)*(node_count-2))

    def path(self,x,y,x2,y2):
        assert self.G!=None
        path = nx.shortest_path(self.G, (x,y), (x2,y2))
        return path


    def get_paths_edges(self, parents, node ):
        queue = deque()
        seen = set()
        queue.append([None,node])
        while queue:
            child, parent = queue.popleft()
            if (child,parent) in seen:
                continue
            seen.add((child,parent))
            yield child,parent
            queue.extend([ (parent,pp) for pp in parents[parent]])


    def all_paths(self, start, end):
        FG=self.FG
        for i in  range(9):
            if start in FG.nodes: 
                break
            x,y=start
            start = (x,y+1)    
        
        parents={start:[]}
            
        queue = deque([(None, start)])
        enqueued,new,seen=set((start,)),set(),set()
        nodes = dict(start=[])

        print("*** next round")
        while queue:
            parent, node=queue.popleft()
            if node==end:
                continue

            nei=set(FG[node])
            new = nei - enqueued
            seen = ( nei & enqueued ) 
            enqueued |= new
            # if not new: # dead end
            #     continue


            for child in new:
                queue.append((node,child))
                assert child not in parents
                parents[child]=[node]
            for s in seen:
                if not s in parents[node]: # if not traversed in opposite direction
                    parents[s].append(node)
            # pprint(queue)
        print ("----- parents:")
        pprint(parents)
        return parents


    @staticmethod
    def assess_connection(G, start, end, max_cut=None):
        ca = connect_assessor_c(G)
        path = nx.shortest_path(G, start,end)
        ca.cycles_along_path(path, max_cut)
        return ca



    def assess_connection_wo_node(self, start, end, max_cut=None):
        TG=self.FG.copy()
        # node_data=self.G.nodes[start]
        # TG.add_node(start, **node_data)
        # TG.add_edges_from(edges)
        ca = connect_assessor_c(TG, self.G, self.OG, self.axes)
        try:
            path = nx.shortest_path(TG, start,end)
        except nx.NetworkXNoPath:
            return None

        ca.cycles_along_path(path, max_cut)
        return ca

    def fake_assess_connection_wo_node(self, start, end, max_cut=None) -> float:
        # TG=self.FG.copy()
        # node_data=self.G.nodes[start]
        # TG.add_node(start, **node_data)
        # TG.add_edges_from(edges)
        lc:float = nx.load_centrality(self.FG, v=start)
        # ca = connect_assessor_c(TG, self.G, self.OG, self.axes)
        # path = nx.shortest_path(TG, start,end)
        # ca.cycles_along_path(path, max_cut)
        return lc

    def check_path(self, start_node:Tuple, end_node:Tuple) -> Optional[List]:
        try:
            FG = self.FG.copy()
            for node in [start_node]:
                node_data = self.G.nodes[node]
                color = node_data['color']
                FG.add_node(node, **node_data)
                for nei_node in self.G[node]:
                    if nei_node in self.FG:
                        FG.add_edge(node, nei_node)
            path=nx.shortest_path(FG,start_node,end_node)
            return path
        except nx.NetworkXNoPath:
            pass
        return None

    def get_components(self):
        comps = nx.connected_components(self.FG)
        component_map=dict()
        nonstuck=set()
        for comp in comps:
            bry = nx.node_boundary(self.G, comp)
            _nonstuck = comp | bry
            for node in _nonstuck:
                component_map.setdefault(node,list()).append((comp, bry))
            nonstuck|=_nonstuck

        # stuck = set(self.G.nodes)-nonstuck
        ns_bry = nx.node_boundary(self.G, nonstuck)
        for stuck_node in ns_bry:
            stuck_node_bry = nx.node_boundary(self.G, [stuck_node])
            component_map.setdefault(stuck_node, list()).append((set(stuck_node), stuck_node_bry))
        return component_map

    def get_bi_components(self):
        comps = nx.biconnected_components(self.FG)
        bi_comp_map=dict()
        for comp in comps:
            for node in comp:
                bi_comp_map.setdefault(node,[]).append(set(comp))
        return bi_comp_map

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
