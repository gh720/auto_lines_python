import networkx as nx
import random,math,copy
import os, sys
from random import randrange,randint
from collections import deque,defaultdict
from attrdict import AttrDict
from pprint import pprint as pp


# PACKAGE_PARENT = '..'
# SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
# sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
# print (SCRIPT_DIR)


from .utils import *

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

    def neighbors(self, node, dir):
        neis = self.G[node]
        neis_sorted = [[None,None] for x in range(4)]
        for i,nei in enumerate(neis):
            if nei==node:
                continue
            nei_dir= direction(node,nei)
            turns = (nei_dir+len(DIRS)-dir)%len(DIRS)
            if turns!=0:
                neis_sorted[turns]=(nei,nei_dir)
        return [ item for item in neis_sorted if item[0]!=None ]

    def right_up(self,pos,dir,shift=0.2):
            pdir = DIRS[prev_i_in_loop(DIRS,dir)]
            return (pos[0]+pdir[0]*shift,pos[1]+pdir[1]*shift)

    def right_down(self,pos,dir,shift=0.2):
            pdir = DIRS[next_i_in_loop(DIRS,dir)]
            return (pos[0]+pdir[0]*shift,pos[1]+pdir[1]*shift)

    def ge(self,edges):
        return [(edge[0], edge[1]) for edge in edges]

    def _sp(self):
        import matplotlib.pyplot as plt



        # import pdb; pdb.set_trace() #DDD 

        dags_by_dir =[x[:] for x in [[]] * 4]
        for edge in self.dagp_edges:
            dags_by_dir[edge[2]].append(edge)

        # dag_right = [ edge for edge in self.dagp_edges if edge[2]==1 ]
        # dag_left = [ edge  for edge in self.dagp_edges if edge[2]==-1 ]

        gna=nx.get_node_attributes

        G=self.G
        # FFG=G.copy()
        DFG=G.copy().to_directed()
        # nx.set_edge_attributes(DFG,1,'capacity')
        plt.figure(1,figsize=(8,8))
        nx.draw(G, pos=gna(G,'pos'))
        #gh = nx.gomory_hu_tree(FFG)
        #uno_cycles = nx.cycle_basis(FFG)
        #pprint(uno_cycles)
        #bg=Board_graph()
        #cycles=Board_graph.change_cycles_direction(uno_cycles,1)
        #for i in range(len(cycles)):
        #    cy=cycles[i]
        #    edges = [ (cy[j],cy[(j+1)%len(cy)]) for j in range(len(cy))]
        #    nx.draw_networkx_edges(DFG,gna(G,'pos'),edgelist=edges,edge_color=colors[i%len(colors)],arrows=True,width=2)
        # me = nx.maximal_matching(FFG)
        
        # nx.set_edge_attributes(DFG)
        for dir,dagp in enumerate(dags_by_dir):
            dag_trunk = [ edge for edge in dagp if self.dagp_edges[edge]==0 ]
            dag_right = [ edge for edge in dagp if self.dagp_edges[edge]==RIGHT ]
            dag_left = [ edge  for edge in dagp if self.dagp_edges[edge]==LEFT ]

            nx.draw_networkx_edges(DFG,pos=gna(G,'pos'), edgelist=self.dagp_edges.keys(), edge_color='b', width=3)
            # nx.draw_networkx_edges(DFG,pos=gna(G,'pos'), edgelist=dag_left, edge_color='b', width=3)
            TG=DFG.edge_subgraph(self.ge(dag_trunk)).copy()
            LG=DFG.edge_subgraph(self.ge(dag_left)).copy()
            RG=DFG.edge_subgraph(self.ge(dag_right)).copy()

            # nx.draw_networkx_edges(DFG,pos=gna(G,'pos'), edgelist=me, edge_color='g', width=2)
            trunk_labels={ (edge[0],edge[1]): self.order.get(edge,'') for edge in dag_trunk }
            left_labels={ (edge[0],edge[1]): self.order.get(edge,'') for edge in dag_left }
            right_labels={ (edge[0],edge[1]): self.order.get(edge,'') for edge in dag_right }
        # pp(gna(G,'pos').items() )
            trunk_pos= gna(G,'pos')#{ n:self.right_up(pos,dir) for n,pos in gna(G,'pos').items() }
            left_pos={ n:self.right_up(pos,dir) for n,pos in gna(G,'pos').items() }
            right_pos ={ n:self.right_down(pos,dir) for n,pos in gna(G,'pos').items() }
            # print(left_pos)
            _=nx.draw_networkx_edge_labels(TG,pos=trunk_pos , label_pos=0.3, edge_labels=trunk_labels, font_color='k')
            _=nx.draw_networkx_edge_labels(LG,pos=left_pos , label_pos=0.3, edge_labels=left_labels, font_color='lime')
            _=nx.draw_networkx_edge_labels(RG,pos=right_pos , label_pos=0.3, edge_labels=right_labels, font_color='r')

        plt.show()


    def cycle_check(self,start_node):
        node_state=dict() # encoutered=1,completed=2
        def dfs(node):
            state=node_state.get(node)
            if state==1:
                assert False # cycle detected
            elif state==None:
                node_state[node]=1
            else:
                return
            parents=self.dagp_rev.get(node)
            if parents:
                for parent in parents:
                    dfs(parent)
            node_state[node]=2
        return


    def update_reachability(self,node):
        parents = self.dagp_rev.get(node)
        if parents:
            reachable=dict()
            for parent in parents:
                prs = self.reachable.get(parent,dict())
                for pr in prs:
                    reachable[pr]=1
            self.reachable[node]=reachable

    def cycles_along_path(self, start_path):
        arc_junctions=self.arc_junctions=dict()
        start =start_path[0]
        path_dict = self.path_dict = self.get_nodes_dict(start_path)
        # queue = deque()
        dagp=self.dagp = defaultdict(list)
        dagp_rev=self.dagp_rev=defaultdict(list)
        dagp_edges=self.dagp_edges=dict()
        trunk_join=self.trunk_join=dict()
        order = self.order=dict()
        reachable = self.reachable=defaultdict(dict)
        counter=0


        def add_to_dag(edge,turn):
            nonlocal counter, self
            parent,node,dir=edge
            if edge in dagp_edges:
                assert False
            # if parent in dagp:
            #     assert False
            if node==(3,6):
                debug=1
            dagp.setdefault(parent,[]).append((node,dir,counter))
            dagp_rev.setdefault(node,[]).append((parent,dir,counter))
            dagp_edges[edge] = turn
            self.update_reachability(node)
            order[edge]=counter
            counter+=1

        # outline=[]
        next_path=[([],None,None)]
        dagp[start_path[-1]]=[]
        for node_i,node in enumerate(start_path):
            parent = None if node_i==0 else start_path[node_i-1]
            dir = None if parent==None else direction(parent,node)
            edge = (parent,node,dir)
            if (parent!=None):
                add_to_dag(edge, turn=0)
            next_path[0][0].append(edge)
            trunk_join[node]=node_i

        current_path=[]
        _counter=9
        all_arcs=self.all_arcs=[]
        iteration=0
        while True:
            if not next_path:
                break
            current_path=copy.deepcopy(next_path)
            next_path=[]
            iteration+=1
            for i, path in enumerate(current_path):
                arc,_trunk_leave_index,arc_turn = path
                if arc_turn==None:
                    arc_turn=0 # main trunk

                for edge_i,edge in enumerate(arc):
                    if counter>=_counter:
                        debug=1
                    parent,node,dir_prev= edge
                    trunk_leave_index = _trunk_leave_index
                    if trunk_leave_index==None:
                        assert iteration==1
                        trunk_leave_index=edge_i

                    edge_turn = dagp_edges.get(edge,0)
                    assert edge_turn==arc_turn
                    dir_prev_back=(dir_prev+2)%len(DIRS) if dir_prev!=None else None
                    # dir_next = direction(node, next_node)
                    dir_next  = None
                    if edge_i < len(arc)-1 :
                        dir_next = direction(node,arc[edge_i+1][1])
                    else:
                        break
                    assert dir_next!=None # end node reached - should have exited loop earlier
                    neis = self.neighbors(node, dir_next)

                    dir_back_suggested=None
                    if dir_prev_back==None: # initial node - taking an arbitrary edge to separate left and right arcs
                        if neis:
                            dir_back_suggested = neis[(len(neis)-1)>>1][1]

                    neis_avl=[]
                    def _scope_218():
                        for next_turn in [LEFT,RIGHT]:
                            if arc_turn!=0 and next_turn!=arc_turn:
                                continue
                            if next_turn==RIGHT:
                                neis.reverse()

                            for nei,nei_dir in neis:
                                # assert turn ==LEFT or turn==RIGHT # arc cannot share an edge with main trunk
                                edge = (node,nei,nei_dir)
                                if nei_dir==dir_prev_back:
                                    break
                                back_dir = (nei_dir + 2) % len(DIRS)
                                back_edge_turn = dagp_edges.get((nei, node, back_dir))

                                skip = False
                                if edge in dagp_edges: # TODO: fix later: skip=True
                                    continue # should not happen - trying to through the edge twice
                                if back_edge_turn!=None:
                                    if back_edge_turn==0: # previous edge on main trunk - we iterated all edges between next and prev clockwise
                                        skip=True
                                    elif back_edge_turn==-next_turn: # that arc was traversed in opposite direction - skip
                                        skip=True
                                    else: # either back_edge_turn==turn or turn==0
                                        neis_avl.append((edge, back_edge_turn))#
                                else:
                                    neis_avl.append((edge, next_turn))
                                if nei_dir==dir_back_suggested:
                                    break

                    _scope_218()

                    def _scope_251():
                        nonlocal next_path
                        for nei_edge,turn in neis_avl:
                            node,nei,dir = nei_edge
                            back_dir=(dir+2)%len(DIRS)
                            back_nei_edge=(nei_edge[1],nei_edge[0],back_dir)
                            # nei_edge = (node,nei,dir)
                            edge_turn = dagp_edges.get(nei_edge,None) # if nei_edge in dagp_edges else None
                            back_edge_turn = dagp_edges.get((nei,node,back_dir),None) # if (nei,node,back_dir) in dagp_edges else None
                            if counter>=_counter:
                                debug=1
                            junction =None
                            junction_ordered=None
                            if edge_turn!=None: # only happens if an edge scheduled for both left and right traversal
                                assert turn==-edge_turn
                                assert back_edge_turn==None
                                junction = arc_junctions.get(nei_edge)
                                junction_ordered = list(reversed(junction)) if junction else None

                            elif back_edge_turn!=None:
                                assert turn==back_edge_turn
                                junction = arc_junctions.get(back_nei_edge,None)
                                junction_ordered = junction if junction else None

                            next_arc=first_junction=last_junction = None
                            if junction_ordered:
                                # if edge_turn!=None:
                                for junc_edge in junction_ordered:
                                    if not junc_edge in dagp_edges:
                                        next_arc,first_junction,last_junction = self.arc_walk(nei_edge, junc_edge, junction, turn, trunk_leave_index)
                                        break
                                # else:
                                #     for junc in reversed(junction):
                                #         if not  junc in dagp_edges:
                                #             next_arc,first_junction,last_junction = self.arc_walk(junc, turn, trunk_leave_index)
                                #             break
                            else:
                                next_arc,first_junction, last_junction = self.arc_walk(nei_edge, None, None,  turn, trunk_leave_index)
                            if next_arc:
                                arc_info=[]
                                first_edge,last_edge=next_arc[0],next_arc[-1]
                                trunk_back_index=trunk_join[last_edge[1]]
                                for i,edge in enumerate(next_arc):
                                    _cnt=counter
                                    arc_info.append((*edge, _cnt))
                                    add_to_dag(edge, turn)
                                    trunk_join[edge[1]]=trunk_back_index
                                if first_junction:
                                    arc_junctions[first_edge]=first_junction
                                    arc_junctions[last_edge]=last_junction

                                self.cycle_check(arc[0][0])
                                next_path+=[[next_arc, trunk_leave_index, turn]]
                                all_arcs.append([arc_info,trunk_leave_index])
                    _scope_251()
        return dagp_edges

    def dfs_walk(self, start_edge, turn):
        queue=deque([start_edge])
        # queue.append(start_edge)
        seen_nodes={start_edge[0]:1}
        arc=[]
        arc_map=dict()
        junction_map=dict()
        first_junction=last_junction=None
        following_earlier_turns=True
        # new_edges=False
        joint_reached = False
        first_edge=None

        def check_turn_consistent(edge):
            nonlocal following_earlier_turns
            edge_back = (edge[1], edge[0], (2 + edge[2]) % len(DIRS))
            turn_fw = self.dagp_edges.get(edge)
            turn_bw = self.dagp_edges.get(edge_back)
            assert turn_fw == None or turn_bw == None
            turn_consistent = False
            if turn_fw != None or turn_bw != None:
                if not following_earlier_turns:
                    return False
                if turn_fw and turn_fw == -turn:
                    turn_consistent = True
                elif turn_bw and turn_bw == turn:
                    turn_consistent = True
                if not turn_consistent:
                    following_earlier_turns = False
                    return False
                return True
            else:
                following_earlier_turns = False
                return True

        while queue:
            edge = queue.pop()
            ok_to_follow = check_turn_consistent(edge)
            if not ok_to_follow:
                continue
            if not following_earlier_turns:
                first_edge = first_edge or edge

            parent,node,dir=edge
            dir_back=(dir+2)%len(DIRS)

            if node in seen_nodes:
                continue
            seen_nodes[node] = 1

            if not following_earlier_turns and node in self.dagp:
                joint_reached=True
                break
            if node in self.path_dict: # all arcs converge on start path
                joint_reached=True
                break


            neis=self.neighbors(node,dir_back)
            neis_it = neis if turn ==RIGHT else neis.reverse()
            for nei,dir in neis:
                nei_edge =(node,nei,dir)
                # nei_edge_back=(nei,node,(2+dir)%len(DIRS))
                # the arc can start with already traversed edges: backwards if with the same turn
                #   or forwards if with opposite turn
                # turn_fw =  self.dagp_edges.get(nei_edge)
                # turn_bw =  self.dagp_edges.get(nei_edge_back)
                # assert turn_fw==None or turn_bw==None
                # turn_consistent = False
                # if turn_fw!=None or turn_bw!=None:
                #     if not following_earlier_turns:
                #         break
                #     if turn_fw and turn_fw==-turn:
                #         turn_consistent=True
                #     elif turn_bw and turn_bw==turn:
                #         turn_consistent=True
                #     if not turn_consistent:
                #         following_earlier_turns=False
                #         break
                # else:
                #     following_earlier_turns=False

                queue.append(nei_edge)
                if not following_earlier_turns:
                    arc_map[nei_edge] = edge

        while queue:
            q = queue.pop()
            junction_map.setdefault(q[0], []).append(q)

        if joint_reached:
            while edge:
                junction = junction_map.get(edge[1])
                if junction:
                    last_junction = last_junction or junction
                    first_junction = junction
                arc.append(edge)
                edge = arc_map.get(edge)
        return arc, first_junction, last_junction

    def arc_walk(self, start_edge, junc_edge, junction, turn, trunk_leave_index):

        assert bool(junc_edge) == bool(junction)
        # edge = junc_edge or start_edge
        edge = start_edge

        arc, first_junction, last_junction = self.dfs_walk(edge,turn)
        if not arc:
            return None,None,None
        if arc[0][-1]==arc[-1][0]:
            return None,None,None

        # if self.trunk_join[last_edge[1]] <= trunk_leave_index:  # return to main path behind our entry
        #     return None, None, None
        reachable = self.reachable.get(arc[-1][0])
        if reachable and reachable.get(arc[0][1]):
            return None,None,None
        return list(reversed(arc)), first_junction, last_junction

    def arc_walk_old(self, start_edge, junc_edge, junction, turn, trunk_leave_index):
        # parent,start_node = edge
        assert bool(junc_edge)==bool(junction)
        edge = junc_edge or start_edge
        prev_node=edge[0]
        node=edge[1]
        inc_dir = edge[2]
        arc=[edge]

        junctions = []
        junction_map=dict()
        if junction:
            junctions.append(junction)

        seen=dict()
        while True:
            if node in self.dagp:
                break
            inc_dir_back = (inc_dir+2)%len(DIRS)
            next_nei=None
            neis = self.neighbors(node, inc_dir_back)

            if turn==RIGHT:
                neis.reverse()

            cut_off_index=None
            next_nei = None
            other_neis = []

            for nei in neis:
                # nei=neis.popleft()
                _t= seen.get((node,nei[0],nei[1]),None)
                if _t: # a loop - cutting it out
                    cut_off_index = _t
                    continue
                _t = seen.get((nei[0],node,nei[1]),None)
                if _t:
                    cut_off_index = _t
                    continue
                if next_nei:
                    other_neis.append(nei)
                else:
                    next_nei=nei
            
            if not next_nei:
                break
            if cut_off_index:
                for edge in arc[cut_off_index:]:
                    jm = junction_map.get((edge[0],edge[1]), junction_map.get((edge[1],edge[0])))
                    if jm!=None:
                        junctions[jm]=None
                arc[cut_off_index:]=[None]*range(cut_off_index,len(arc))

            next_edge= (node,next_nei[0],next_nei[1])

            arc.append(next_edge)
            if other_neis:
                for nei,nei_dir in other_neis:
                    junctions.append((node,nei,nei_dir))
                    junction_map[(node,nei)]=len(junctions)-1
                # last_junction=[ (node,nei,nei_dir) for nei,nei_dir in other_neis]
                # first_junction= first_junction or last_junction

            seen[next_edge]=len(arc)-1
            inc_dir=next_nei[1]
            prev_node=node
            node=next_nei[0]

        if not node in self.dagp: # dead end
            return None,None,None
        
        if self.trunk_join[node] <= trunk_leave_index: # return to main path behind our entry
            return None,None,None

        arc=[ item for item in arc if item !=None]

        return arc,first_junction, last_junction


    def assess_path(self,path):
        G=self.G
        for i,node in enumerate(path):
            if i==len(path):
                break
            neis= G[node]
            for nei in neis:
                if nei==path[i+1]:
                    continue
                



