import networkx as nx
import random,math,copy
import os, sys
from random import randrange,randint
from collections import deque,defaultdict
from attrdict import AttrDict


# PACKAGE_PARENT = '..'
# SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
# sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
# print (SCRIPT_DIR)


from utils import *

# constants

'''
for p in initial_path
    add_to_dag(p)   
current_path=initial_path
for p in current_path
    queue_branches(p)
current_path=[]
for b in branches
    build cycle
    for c in cycle
        add_to_dag(c)
    add_to_current_path

-------

for p in initial_path
    add_to_dag(p)   
current_path=initial_path
for p in current_path
    for b in branches(p)
        build cycle
        if cycle arrives at farther point in outline
            for c in cycle
                add_to_dag(c)

'''

class connect_assessor_c:

    cycle_sets=AttrDict()

    def __init__(self,G):
        self.G=G

    def assess_connection(self,start,end):
        self.path = nx.shortest_path(start,end)
        self.assess_path(self.path)

    def check_direction_(selfcycle_sets,cy_i,node,next_node):
        nm=cycle_sets.node_map
        cn_i=nm[node][cy_i] if node in nm else None
        cn_i_next=nm[next_node][cy_i] if next_node in nm else None
        if next_i(cycle_sets.cycles[cy_i],cn_i)==cn_i_next:
            return 1
        if prev_i(cycle_sets.cycles[cy_i],cn_i)==cn_i_next:
            return -1
        return None

    def check_turn_direction(self, a,b,c):
        assert False


    def get_nodes_dict(self, nodes):
        d=  { node:i for i,node in enumerate(nodes) }
        return d

    def get_neighbors(self, node, dir):
        neis = self.G.nodes[node]
        neis_sorted = [[None,None] for x in range(4)]
        for i,nei in enumerate(neis):
            if nei==node:
                continue
            nei_dir= get_direction(node,nei)
            turns = (nei_dir+len(DIRS)-dir)%len(DIRS)
            neis_sorted[turns]=(nei,nei_dir)
        return [ item for item in neis_sorted if item[0]!=None ]


    def cycles_along_path(self, start_path):
        junctions=self.junctions=dict()
        start =start_path[0]
        path_dict = self.get_nodes_dict(start_path)
        # queue = deque()
        dagp=self.dagp = defaultdict(list)
        dagp_edges=self.dagp_edges=dict()
        trunk_join=dict()
        order = self.order=dict()
        counter=0



        def add_to_dag(edge,turn):
            parent,node,dir=edge
            dagp.setdefault(parent,[]).append((node,dir,counter))
            dagp_edges[edge]=turn
            order[node]=counter
            counter+=1

        # outline=[]
        next_path=[[],None]
        for node_i,node in enumerate(start_path):
            parent = None if node_i==0 else start_path[node_i-1]
            add_to_dag(edge=(parent,node ), turn=0)
            next_path[0].append(parent,node, 0 if parent==None else get_direction(parent,node))
            trunk_join[node]=node_i

        current_path=[]
        while True:
            if not next_path:
                break
            current_path=next_path
            next_path=[]
            for i, path in enumerate(current_path):
                arc,trunc_leave_index,turn = path
                if trunc_leave_index==None: # for main trunk then entry is a node index
                    trunk_entry=i
                for edge_i,edge in enumerate(arc):
                    parent,node,dir_prev= edge
                    # dir_next = get_direction(node, next_node)
                    dir_next  = None
                    if edge_i < len(arc)-1 :
                        dir_next = get_direction(node,arc[edge_i+1][1])
                    assert dir_next!=None # end node reached - should have exited loop earlier
                    neis = self.get_neighbors(node,dir_next) 

                    if dir_prev==None: # initial node - taking an arbitrary edge to separate left and right arcs
                        if neis:
                            dir_prev = neis[(len(neis)-1)>>1][1]

                    for next_turn in [LEFT,RIGHT]:
                        if next_turn==RIGHT:
                            neis.reverse()

                        for nei,nei_dir in neis:
                            # assert turn ==LEFT or turn==RIGHT # arc cannot share an edge with main trunk
                            edge = (node,nei,nei_dir)
                            if dagp_edges[edge]:
                                assert False # shoult not happen - trying to through the edge twice
                            back_edge_turn = dagp_edges[(nei,node)]
                            if back_edge_turn!=None:
                                if back_edge_turn==0: # previous edge on main trunk - we iterated all edges between next and prev clockwise 
                                    continue
                                elif back_edge_turn==-turn: # that arc was traversed in opposite direction - skip
                                    continue
                                else: # either back_edge_turn==turn or turn==0
                                    neis_avl.append((edge, back_edge_turn))# 
                            else:
                                neis_avl.append((edge, next_turn))
                            if nei_dir==dir_prev:
                                break
                    
                    for nei,dir,turn in neis_avl:
                        nei_edge = (node,nei,dir)
                        edge_turn = dagp_edges(nei_edge)
                        back_edge_turn = dagp_edges((nei,node))
                        if edge_turn!=None: # only happens if an edge scheduled for both left and right traversal
                            assert turn==-edge_turn
                            assert back_edge_turn==None
                            junction = arc_junctions[nei_edge]
                        elif back_edge_turn!=None:
                            assert turn==back_edge_turn
                            junction = arc_junctions[nei_edge]

                        if junction:
                            if turn==LEFT:
                                for junc in junctions:
                                    if dagp_edges[junc]==None:
                                        next_arc,first_junction,last_junction = self.arc_walk(junc, turn, trunk_leave_index)
                                        break
                            else:
                                for junc in reversed(junctions):
                                    if dagp_edges[junc]==None:
                                        next_arc,first_junction,last_junction = self.arc_walk(junc, turn, trunk_leave_index)
                                        break
                        else:
                            next_arc,first_junction, last_junction = self.arc_walk(nei_edge, turn)
                        if next_arc:
                            first_edge,last_edge=next_arc[0],next_arc[-1]
                            trunk_back_index=trunk_join[last_edge[1]]
                            for i,edge in enumerate(next_arc):
                                add_to_dag(edge, turn)
                                trunk_join[edge[1]]=trunk_back_index
                            if first_junction:
                                junctions[first_edge]=first_junction
                                junctions[last_edge]=last_junction

                        next_path+=[next_arc, trunk_leave_index]


    def arc_walk(self,edge,turn, trunk_leave_index):
        # parent,start_node = edge
        prev_node=edge[0]
        node=edge[1]
        inc_dir = edge[2]
        
        arc=[prev_node]
        
        last_junction=[]
        first_junction=[]
        seen=set()
        while True:
            inc_dir_back = (inc_dir+2)%len(DIRS)
            next_nei=None
            neis = self.get_neighbors(node,inc_dir_back)

            cut_off_index=None
            if not neis:
                break
            next_nei = None
            next_neis = []
            while True:
                nei=neis.popleft()
                if not nei: 
                    break
                _t= seen[(node,nei[0],nei[1])]
                if _t: # a loop - cutting it out
                    cut_off_index = _t
                    continue
                _t = seen[(nei[0],node,nei[1])]
                if _t:
                    cut_off_index = _t
                    continue
                if next_nei:
                    next_neis.append(nei)
                else:
                    next_nei=nei
            
            if not next_nei:
                return None
            if cut_off_index:
                arc[cut_off_index:]=[None]*range(cut_off_index,len(arc))
            next_edge= (node,next_nei[0],next_nei[1])
            if next_neis:
                last_junction=[ (node,nei,nei_dir) for nei,nei_dir in next_neis]
                first_junction= first_junction or last_junction 

            arc.append(next_edge)

            if next_nei in dagp:
                break
            prev_node=node
            node=next_nei

        if next_nei==None: # deadend
            return None
        
        if trunk_join[next_nei] <= trunk_le: # return to main path behind our entry
            return None

        arc=[ item for item in arc if item !=None]
        return arc,first_junction, last_junction



        while next_iter:
            while queue:
                node, turn, prev_dir, parent = queue.popleft()
                node_x,node_y = node

                neis = [ n for n in G[node] if n!=parent  ]
                dir2nei = { get_direction(node,nei):nei for nei in neis }
                nei2dir = { y:x for x,y in shift2nei.items() }

                if neis:
                    if turn!=None:
                        assert prev_dir!=None and parent!=None
                        
                        while True:
                            dir = DIRS[(prev_dir+TURN)%len(DIRS)]
                            nei = dir2nei[dir]
                            if not nei:
                                continue
                            assert dir!=prev_dir
                            queue.append(nei, turn, dir, node)
                            dagp[nei].append(node)
                            for _nei in neis:
                                if _nei!=nei:
                                    next_iter.append[(node,turn,nei2dir(_nei),parent)]
                    else:
                        for nei in neis:
                            if nei in seen:
                                continue # STUB
                            queue.append((nei,LEFT,nei2dir[nei],node))
                            queue.append((nei,RIGHT,nei2dir[nei],node))
                            dagp[nei].append(node)


    def assess_path(self,path):
        G=self.G
        for i,node in enumerate(path):
            if i==len(path):
                break
            neis= G[node]
            for nei in neis:
                if nei==path[i+1]:
                    continue
                



