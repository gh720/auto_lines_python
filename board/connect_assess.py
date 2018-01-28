
from sched import scheduler

import networkx as nx
import random,math,copy
import os, sys
from random import randrange,randint
from collections import deque,defaultdict
from attrdict import AttrDict
import pprint
from pprint import pprint as pp
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import bisect
import re


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


def back_edge(edge):
    return (edge[1], edge[0], (edge[2] + 2) % len(DIRS))

class connect_assessor_c:
    start_node=None
    end_node=None
    cycle_sets=AttrDict()
    debug_edge = ((5, 7), (4, 7), 3)
    debug_edges = [debug_edge, back_edge(debug_edge)]
    debug_face = ((4, 4), (3, 4), 3)
    debug_face2 = ((None), (None), 1)

    # outer_node = ((None, None))
    outer_face = 'outer'
    skipped_faces=[]
    edges_to_skip=dict()
    cycles_skipping=[]

    def __init__(self,G,posG=None,occG=None,axes=None):
        self.axes=axes
        self.posG=posG
        self.G=G
        self.occG=occG
        self.DAG=nx.DiGraph()

    def assess_connection(self,start,end):
        self.path = nx.shortest_path(start,end)
        self.assess_path(self.path)

    def next_face_name(self, prefix=None):
        if prefix==None:
            prefix='skipped_'
        return prefix + str(len(self.faces))

    def get_nodes_dict(self, nodes):
        d=  { node:i for i,node in enumerate(nodes) }
        return d

    def get_edges_dict(self, nodes):
        d=  { (*e, direction(e[0],e[1])):1 for e in (
                [tuple(nodes[i-1:i+1]) for i in range(1,len(nodes))]
                )}
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

    def _sd(self, block=False):
        import matplotlib.pyplot as plt

        G = self.dualG
        plt.figure(1, figsize=(6, 6))
        pos = nx.shell_layout(G) # nx.kamada_kawai_layout(G) # nx.fruchterman_reingold_layout(G) #.spring_layout(G)
        nx.draw(G, pos=pos)
        labels = { node: pprint.pformat((node[0],node[1]))  for node in G.nodes }
        _ = nx.draw_networkx_labels(G, pos=pos, labels=labels, font_color='k', font_size=8)
        plt.show(block=block)

    def _sp(self,axes=False, block=False):
        # import matplotlib.pyplot as plt
        if axes == True:
            axes=self.axes
        pp("axes:" + str(axes))
        if axes:
            plt.sca(axes)

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
        plt.figure(1,figsize=(7,7))
        # plt.subplot(111)
        nx.draw_networkx(G, pos=gna(self.posG,'pos'), node_shape='s', node_size=50, with_labels=False)
        if self.occG:
            nx.draw_networkx(self.occG, pos=gna(self.posG,'pos'), node_shape='o'
                    , node_color=[ mpl.colors.cnames[y] for x,y in gna(self.occG,'color').items()]
                    , node_size=500, with_labels=False)
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

        # return
        face_starting_edges=[]
        for face in self.faces:
            edge = self.initial_edge(face)
            if edge:
                face_starting_edges.append((edge, 'lime' if self.face_turn_wrt_edge(face,edge)==LEFT else 'r') )
        # face_starting_edges = [ (self.e2fac, 'lime' if self.faces[face]['turn']==LEFT else 'r' )
        #                             for face in self.faces
        #                                 if face!=self.outer_face and not re.match('^skipped',str(face))]
        # de_parents=dict()
        # de_children=dict()
        edge2dual={ de[2]:de for de in self.dual_edges }
        # for de in self.dual_edges:
        #     edge2dual[de[2]]=de
            # de_parents.setdefault(de[1], dict())[de[0]]=de[2]
            # de_children.setdefault(de[0], dict())[de[1]]=de[2]

        de_edges_by_dir = [x[:] for x in [[]] * 4]

        for edge,edge_turn in self.dagp_edges.items():
            face=listget(0,self.e2face.get(edge))
            # if not self.faces[face]['edges'][edge]:
            #     assert False
            if not face: continue

            de=edge2dual.get(edge)
            if not de:
                continue

            f0_turn = self.face_turn_wrt_edge(de[0], edge)
            f1_turn = self.face_turn_wrt_edge(de[1], edge)
            f0_turn = f0_turn or -f1_turn
            f1_turn = f1_turn or -f0_turn

            rotate=f1_turn

            de_edges_by_dir[edge[2]].append((edge, rotate))

        for dir,edges in enumerate(de_edges_by_dir):
            de_right = [ edge for edge,rotate in edges if rotate==RIGHT ]
            de_left = [ edge  for edge,rotate in edges if rotate==LEFT ]

            DLG=DFG.edge_subgraph(self.ge(de_left)).copy()
            DRG=DFG.edge_subgraph(self.ge(de_right)).copy()

            # nx.draw_networkx_edges(DFG,pos=gna(G,'pos'), edgelist=me, edge_color='g', width=2)
            left_labels={ (edge[0],edge[1]): '*' for edge in de_left }
            right_labels={ (edge[0],edge[1]): '*' for edge in de_right }
        # pp(gna(G,'pos').items() )
            left_pos={ n:self.right_up(pos,dir) for n,pos in gna(G,'pos').items() }
            right_pos ={ n:self.right_down(pos,dir) for n,pos in gna(G,'pos').items() }
            # _=nx.draw_networkx_edge_labels(DLG,pos=left_pos , label_pos=0.5, edge_labels=left_labels, font_size=8, font_color='lime')
            # _=nx.draw_networkx_edge_labels(DRG,pos=right_pos , label_pos=0.5, edge_labels=right_labels, font_size=8, font_color='r')

        nx.draw_networkx_edges(DFG,pos=gna(G,'pos')
                               , edgelist=[ v[0] for v in face_starting_edges ]
                               , edge_color=[ v[1] for v in face_starting_edges ]
                               , width=2.0)



        for dir,dagp in enumerate(dags_by_dir):
            dag_trunk = [ edge for edge in dagp if self.dagp_edges[edge]==0 ]
            dag_right = [ edge for edge in dagp if self.dagp_edges[edge]==RIGHT ]
            dag_left = [ edge  for edge in dagp if self.dagp_edges[edge]==LEFT ]

            TG = DFG.edge_subgraph(self.ge(dag_trunk)).copy()
            LG = DFG.edge_subgraph(self.ge(dag_left)).copy()
            RG = DFG.edge_subgraph(self.ge(dag_right)).copy()

            # nx.draw_networkx_edges(DFG,pos=gna(G,'pos'), edgelist=me, edge_color='g', width=2)
            trunk_labels = {(edge[0], edge[1]): self.order.get(edge, '') for edge in dag_trunk}
            left_labels = {(edge[0], edge[1]): self.order.get(edge, '') for edge in dag_left}
            right_labels = {(edge[0], edge[1]): self.order.get(edge, '') for edge in dag_right}
            # pp(gna(G,'pos').items() )
            trunk_pos = gna(G, 'pos')  # { n:self.right_up(pos,dir) for n,pos in gna(G,'pos').items() }
            left_pos = {n: self.right_up(pos, dir) for n, pos in gna(G, 'pos').items()}
            right_pos = {n: self.right_down(pos, dir) for n, pos in gna(G, 'pos').items()}
            _ = nx.draw_networkx_edge_labels(TG, pos=trunk_pos, label_pos=0.3, font_size=8, edge_labels=trunk_labels, font_color='k')
            # _ = nx.draw_networkx_edge_labels(LG, pos=left_pos, label_pos=0.3, edge_labels=left_labels,
            #                                  font_color='lime')
            # _ = nx.draw_networkx_edge_labels(RG, pos=right_pos, label_pos=0.3, edge_labels=right_labels, font_color='r')

            nx.draw_networkx_edges(DFG,pos=gna(G,'pos'), edgelist=self.dagp_edges.keys(), edge_color='b', width=3)
            # nx.draw_networkx_edges(DFG,pos=gna(G,'pos'), edgelist=dag_left, edge_color='b', width=3)



        plt.show(block=block)


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
                    dfs(parent[0])
            node_state[node]=2
        dfs(start_node)
        return


    def update_reachability(self,node):
        reachable_parents={node:1}
        reachable_children={node:1}
        parents = self.dagp_rev.get(node)
        if parents:
            for parent in parents:
                prs = self.reachable_parents.get(parent[0], dict({parent[0]:1}))
                for pr in prs:
                    reachable_parents[pr] = 1
        children = self.dagp.get(node)
        if children:
            for child in children:
                chs = self.reachable_children.get(child[0], dict({child[0]:1}))
                for ch in chs:
                    reachable_children[ch] = 1
        for pr in reachable_parents:
            for ch in reachable_children:
                self.reachable_parents.setdefault(ch,dict())[pr]=1
                self.reachable_children.setdefault(pr, dict())[ch] = 1

    def add_to_dag(self,edge,turn):
        parent,node,dir=edge
        if edge in self.dagp_edges:
            assert False
        if parent!=None:
            self.dagp.setdefault(parent,[]).append((node,dir,self.counter))
            self.dagp_rev.setdefault(node,[]).append((parent,dir,self.counter))
            self.dagp_edges[edge] = turn
            self.order[edge] = self.counter
            self.counter += 1
        self.update_reachability(node)


    def cycles_along_path(self, start_path, max_node_cut=None):
        arc_junctions=self.arc_junctions=dict()
        # start =start_path[0]
        self.start_node=start_path[0]
        self.end_node=start_path[-1]
        self.max_node_cut=max_node_cut
        self.path_dict = self.get_nodes_dict(start_path)
        self.path_edge_dict= self.get_edges_dict(start_path)
        self.dagp = defaultdict(list)
        self.dagp_rev=defaultdict(list)
        self.dagp_edges=dict()
        trunk_join=self.trunk_join=dict()
        self.order=dict()
        self.reachable_parents=dict()
        self.reachable_children=dict()
        counter=self.counter=0
        self.e2face=dict()
        self.faces=dict()
        self.dual_edges=dict()
        self.outer_cycle=[]
        self.outer_edges=dict()

        trunk_turn =0

        self.dagp[start_path[-1]]=[]
        expansion = ([],trunk_turn,dict())
        for node_i,node in enumerate(start_path):
            parent = None if node_i==0 else start_path[node_i-1]
            dir = None if parent==None else direction(parent,node)
            edge = (parent,node,dir)
            self.add_to_dag(edge, trunk_turn)
            junction = []
            if node_i < len(start_path)-1:
                neis = self.neighbors(node,direction(node,start_path[node_i+1]))
                junction = [ (node, *nei) for nei in neis if back_edge((node, *nei)) not in self.dagp_edges ]
            expansion[0].append((edge,junction))
            trunk_join[node]=node_i

        arc_queue=deque([expansion])

        iteration=0
        scheduled = deque()
        scheduled_late = list()
        while arc_queue:
            expansion = arc_queue.popleft()
            iteration+=1
            if iteration==17:
                debug=1
            arc,arc_turn,meta = expansion

            if arc:
                arc_info = []
                for arc_i, item in enumerate(arc):
                    edge, _junction=item
                    _cnt = counter
                    arc_info.append((*edge, _cnt))

                    bw = back_edge(edge)[2] if edge[0]!=None else None
                    fw = arc[arc_i+1][0][2] if arc_i < len(arc)-1 else None
                    if fw == None:
                        continue # last arc edge - don't need junctions
                    junction = []
                    if _junction != None:
                        for junc_edge in _junction:
                            if junc_edge == ((5, 7), (4, 7), 3):
                                debug = 1
                            if junc_edge in self.dagp_edges:
                                continue
                            if back_edge(junc_edge) in self.dagp_edges:
                                continue
                            junction.append(junc_edge)
                        left_branches, right_branches = self.balanced_turn(junction, fw, bw)

                        # for br in left_branches:
                        #     scheduled.append((br, LEFT))
                        # for br in reversed(right_branches):
                        #     scheduled.append((br, RIGHT))

                        for br_i in range(max(len(left_branches),len(right_branches))):
                            if br_i < len(left_branches):
                                assert arc_turn!=RIGHT
                                scheduled.append((left_branches[br_i], LEFT))
                            if br_i < len(right_branches):
                                assert arc_turn!=LEFT
                                scheduled.append((right_branches[br_i], RIGHT))

            # expansions=[]

                def process_item(item):
                    edge, suggested_turn = item
                    if edge in self.dagp_edges or back_edge(edge) in self.dagp_edges:
                        return
                    tup, next_arc, arc_back, sum_of_turns, cycle_found, *_ = [None] * 10
                    for edge_turn in [suggested_turn, -suggested_turn]:
                        outer_turn = self.edges_to_skip.get(edge)
                        if outer_turn == edge_turn:  # if this will traverse outer cycle retry with opposite turn
                            continue
                        tup = self.arc_walk(edge, edge_turn)
                        if not type(tup) is tuple:
                            if tup == 'noose':
                                break
                            if tup == 'cycle':
                                cycle_found = True
                                continue
                            if tup == 'cycle_part':
                                continue
                            break  # other cases, whatever
                        _next_arc, arc_back, sum_of_turns = tup
                        if _next_arc[0][0]==((2,2),(3,2),1):
                            debug=1
                        if not sum_of_turns:
                            assert False
                        if (sum_of_turns > 0 and edge_turn == LEFT
                                or sum_of_turns < 0 and edge_turn == RIGHT):
                            self.add_edges_to_skip(_next_arc + arc_back, edge_turn)
                            continue

                        # check if backwards part of a cycle can make a loop
                        def check_for_cycle_395():
                            stretches=[]
                            start_node=end_node=None
                            for arc_edge, _ in arc_back:
                                if arc_edge not in self.dagp_edges and back_edge(arc_edge) not in self.dagp_edges:
                                    start_node= start_node or arc_edge[0]
                                if arc_edge[1] in self.dagp:
                                    if start_node:
                                        stretches.append([start_node, arc_edge[1]])
                                    start_node=None
                            for i in range(len(stretches)):
                                for j in range(i,len(stretches)):
                                    start_node=stretches[i][0]
                                    end_node=stretches[j][1]
                                    reachable = self.reachable_parents.get(start_node)
                                    if reachable and reachable.get(end_node):
                                        break
                                else:
                                    continue
                                break
                            else:
                                return False # cycle not found
                            return True
                        if check_for_cycle_395():
                            cycle_found=True
                            break
                        next_arc = _next_arc
                        break # normal arc

                    if not next_arc:
                        if cycle_found:
                            scheduled_late.append((edge, suggested_turn))
                        return

                    face_turn = RIGHT if sum_of_turns > 0 else LEFT
                    # face_turn_wrt_start_edge=None

                    for arc_edge, junction in next_arc:
                        self.add_to_dag(arc_edge, edge_turn)

                    for de in self.debug_edges:
                        if (de in [ e[0] for e in next_arc]
                                or de in [ e[0] for e in arc_back]) :
                            debug=1
                    self.add_face(None, next_arc, arc_back, face_turn)

                    # for peer_face, face_dir in peer_faces.items():
                    #     if face_dir == 'from':
                    #         dual_edges[(peer_face, arc_face)] = 1
                    #     else:
                    #         dual_edges[(arc_face, peer_face)] = 1

                    if next_arc:
                        self.cycle_check(next_arc[0][0][1])
                        arc_queue.append((next_arc, edge_turn, dict()))

                while scheduled:
                    item = scheduled.popleft()
                    process_item(item)

        while scheduled_late:
            next_try=copy.copy(scheduled_late)
            scheduled_late=[]
            for item_i in range(len(next_try)-1,-1,-1):
                process_item(next_try[item_i])
            if len(next_try)<=len(scheduled_late):
                # self._sp()
                # assert False
                break

        self.make_nx_dag()

        self.check_missed_faces()

        if not self.create_dual_graph():
            assert False

        self.compute_path_cuts()
        self.compute_node_cuts()

        return

######### DAG

    def make_nx_dag(self):
        self.DAG=nx.DiGraph()
        self.DAG.add_edges_from([
            (edge[0],edge[1]) for edge in self.dagp_edges
        ])

    def get_edge_w_dir(self,edge):
        return (edge[0],edge[1], direction(edge[0],edge[1]))

    def get_edge(self,edge):
        return (edge[0], edge[1])



    def add_edges_to_skip(self, cycle, turn):
        self.cycles_skipping.append(cycle)
        for outer_edge, junction in cycle:
            if (outer_edge not in self.dagp_edges
                and back_edge(outer_edge) not in self.dagp_edges):
                self.edges_to_skip[outer_edge] = turn
                self.edges_to_skip[back_edge(outer_edge)] = -turn


    def balanced_turn(self, junctions, fw, bw):
        assert fw!=None
        assert fw!=bw
        left=[]
        right=[]

        jar = [None]*4
        for i,junc in enumerate(junctions):
            jar[junc[2]]=junc
            assert fw!=junc[2]
            assert bw!=junc[2]

        if bw !=None and fw !=None:
            bw_seen = False
            dir = fw
            while True:
                dir = ( dir + 1 )% len (DIRS)
                if dir == fw:
                    break
                if dir == bw:
                    bw_seen=True
                if jar[dir]:
                    if bw_seen:
                        right.insert(0,jar[dir])
                    else:
                        left.append(jar[dir])
        elif bw==None:
            # counter=0
            dir = fw
            bw_sugg = (fw + 2) % len(DIRS)
            bw_seen = False
            while True:
                dir = (dir + 1) % len(DIRS)
                if dir == fw:
                    break
                if dir == bw_sugg:
                    bw_seen = True
                if jar[dir]:
                    if not bw_seen:
                    # if counter<(len(junctions)>>1):
                        left.append(jar[dir])
                    else:
                        right.insert(0,jar[dir])
                    # counter+=1
        elif fw == None:
            # counter = 0
            dir = bw
            fw_sugg = (bw + 2) % len(DIRS)
            fw_seen = False
            while True:
                dir = (dir + 1) % len(DIRS)
                if dir == bw:
                    break
                if dir == fw_sugg:
                    fw_seen = True
                if jar[dir]:
                    if fw_seen:
                    # if counter >= (len(junctions) >> 1):
                        left.append(jar[dir])
                    else:
                        right.insert(0, jar[dir])
                    # counter += 1

        return left, right

    def dfs_walk_turning_undiscovered(self, start_edge, turn):
        queue=deque([start_edge])
        # queue.append(start_edge)
        # seen_nodes={start_edge[0]:1}
        seen_nodes=dict()
        arc=[]
        arc_map=dict()
        junction_map=dict()
        first_junction=last_junction=None
        # following_earlier_turns=True
        following_earlier_turns=False # no more
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


            if not following_earlier_turns and node in self.dagp:
                joint_reached=True
                break
            if node in self.path_dict:
                joint_reached=True
                break

            if node in seen_nodes: # late checking, so cycles are returned, but "nooses" aren't
                continue
            seen_nodes[node] = 1

            neis=self.neighbors(node,dir_back)
            neis_it = neis if turn ==RIGHT else neis.reverse()
            for nei,dir in neis:
                nei_edge =(node,nei,dir)
                queue.append(nei_edge)
                if not following_earlier_turns:
                    arc_map[nei_edge] = edge

        while queue:
            q = queue.pop()
            junction_map.setdefault(q[0], []).append(q)

        if not joint_reached:
            return []
        while edge:
            junction = junction_map.get(edge[1])
            arc.append((edge,junction))
            edge = arc_map.get(edge)
        return arc

    def dfs_walk_turning_back(self, start_edge, end_node, turn):
        queue = deque([start_edge])
        arc = []
        arc_map = dict()
        seen_nodes=dict()
        while queue:
            edge = queue.pop()
            parent, node, dir = edge
            dir_back = back_edge(edge)[2]

            if node in seen_nodes:
                continue

            seen_nodes[node] = 1

            if node==end_node:
                break

            neis = self.neighbors(node, dir_back)
            neis_it = neis if turn == RIGHT else neis.reverse()
            for nei, dir in neis:
                nei_edge = (node, nei, dir)
                queue.append(nei_edge)
                arc_map[nei_edge] = edge
        else:
            return arc # no cycle found

        while edge:
            arc.append((edge, None))
            edge = arc_map.get(edge)
        return arc

    def arc_walk(self, start_edge, turn):
        arc  = self.dfs_walk_turning_undiscovered(start_edge, turn)
        if not arc:
            return 'noose'  #  dead end
        if arc[0][0][1]==arc[-1][0][0]:
            return 'cycle'  #  cycle


        reachable = self.reachable_parents.get(arc[-1][0][0]) # BETTER: classes instead of tuples
        if reachable and reachable.get(arc[0][0][1]):
            return 'cycle_part' # forms a cycle

        arc_back = self.dfs_walk_turning_back(arc[0][0], start_edge[0], turn)
        arc_back.pop() # because arc[0][0] included

        assert bool(arc_back)

        sum_of_turns=-self.sum_of_turns(arc_back+arc)
        return (list(reversed(arc)), list(reversed(arc_back)), sum_of_turns)

######### faces

    def create_dual_graph(self):
        edge2dual = {de[2]: de for de in self.dual_edges}
        self.dualG = nx.MultiDiGraph()
        for edge in self.dual_edges:
            self.dualG.add_edge(edge[0],edge[1], planar_edge= edge[2])

        for edge, edge_turn in self.dagp_edges.items():
            face = listget(0,self.e2face.get(edge))
            if not face: continue
            de = edge2dual.get(edge)
            f0_turn = self.face_turn_wrt_edge(de[0], edge)
            f1_turn = self.face_turn_wrt_edge(de[1], edge)
            f0_turn = f0_turn or -f1_turn
            f1_turn = f1_turn or -f0_turn

            rotate=f1_turn

            if rotate!=RIGHT:
                if f0_turn==0 and f1_turn==0 and de[0]=='outer' and de[1]=='outer':
                    return True
                return False
        return True

    def initial_edge(self,face):
        edges=self.faces[face]['edges']
        for edge in edges:
            if self.e2face.get(edge,[None,None])[0]==face:
                return edge
        return None

    def add_face(self,arc_face,next_arc,arc_back,face_turn):
        for arc_edge, junction in next_arc:
            if not arc_face:
                arc_face = arc_edge  # face labeled after the edge that formed it
                self.faces[arc_face] = {'edges': {arc_edge: face_turn}, 'turn': face_turn}
            else:
                self.faces[arc_face]['edges'][arc_edge] = face_turn
            self.e2face.setdefault(arc_edge, [None, None])[0] = arc_face

        assert arc_face
        # if arc_face=='outer' and  arc_face in self.faces:
        #     debug=1

        if arc_face not in self.faces:  # sanity check
            self.faces[arc_face] = {'edges': dict(), 'turn': face_turn}

        for arc_edge, junction in arc_back:
            fw_face = listget(0, self.e2face.get(arc_edge))
            bw_face = listget(0, self.e2face.get(back_edge(arc_edge)))
            if bool(fw_face) and bool(bw_face):
                assert False
            if not fw_face and not bw_face:
                for edge in (arc_edge, back_edge(arc_edge)):
                    edge_turn = self.dagp_edges.get(edge)  # edges of the main path treated as new edges of the cycle
                    if edge_turn==0: # path edge
                        self.faces[arc_face]['edges'][edge] = (
                                face_turn if edge==arc_edge else -face_turn
                        )
                        if arc_face!='outer':
                            assert edge!=arc_edge # path edges are always backwards
                        # self.faces[arc_face].setdefault('initial', edge)
                        self.e2face.setdefault(edge, [None, None])[0] = arc_face
                        break
                    elif edge_turn!=None: # that is the edge exists
                        self.faces[arc_face]['edges'][edge] = (
                            face_turn if edge==arc_edge else -face_turn
                        )
                        # self.faces[arc_face].setdefault('initial', arc_edge)
                        self.e2face.setdefault(arc_edge, [None, None])[0] = arc_face
                        break
                continue
            de_edge = None
            if fw_face:
                # peer_faces[fw_face] = 'from' if face_turn == RIGHT else 'to'
                de_edge = (fw_face, arc_face, arc_edge) if face_turn == RIGHT else (arc_face, fw_face, arc_edge)
                self.faces[arc_face]['edges'][arc_edge] = face_turn
                self.e2face.setdefault(arc_edge, [None,None])[1] = arc_face
            elif bw_face:
                # peer_faces[bw_face] = 'from' if face_turn == LEFT else 'to'
                de_edge = (bw_face, arc_face, back_edge(arc_edge)) if face_turn == LEFT else (
                arc_face, bw_face, back_edge(arc_edge))
                self.faces[arc_face]['edges'][back_edge(arc_edge)] = -face_turn
                self.e2face.setdefault(back_edge(arc_edge), [None, None])[1] = arc_face

            assert de_edge
            self.dual_edges[de_edge] = 1



    def face_turn_wrt_edge(self,face,edge):
        return self.faces[face]['edges'][edge]

    def face_turn_wrt_edge_old(self,face,edge):
        if face=='outer':
            faces= self.e2face.get(edge)
            if not faces:
                return 0
            assert faces[1] =='outer'
            if faces[0] == 'outer':
                return 0
            turn = -self.faces[faces[0]]['turn'] # opposite of adjacent face turn
        else:
            turn = self.faces[face]['turn']
        if edge in self.path_edge_dict: # path edges backwards
            turn =-turn
        return turn

    # add missed faces: the outer face and faces created by a chord
    def check_missed_faces(self):
        items=[]
        for edge, edge_turn in self.dagp_edges.items():
            faces = self.e2face.get(edge)
            if not faces:
                items.append((edge,edge_turn)) # late
            else:
                items.insert(0,(edge,edge_turn)) # early

        for edge,edge_turn in items:
            faces=self.e2face.get(edge)
            if not faces: # bridge edge - outer face on both sides
                # sanity check - a bridge can't make a cycle
                for turn in LEFT,RIGHT:
                    cycle=self.dfs_walk_turning_back(edge, edge[0], turn)
                    if cycle:
                        assert False
                if edge[0] not in self.path_dict or edge[1] not in self.path_dict:
                    assert False
                # face=self.next_face_name('bridge_cut_')
                face='outer'
                if not face in self.faces:
                    self.add_face(face, [], [(edge,None)], 0)
                self.e2face[edge]=[face,face]
                self.faces[face]['edges'][edge]=0 # no turn for bridge edge
                de_edge = (face, face, edge)
                self.dual_edges[de_edge]=1
                continue
            if len(faces)!=2:
                assert False
            if bool(faces[0])!=bool(faces[1]):
                peer_face=faces[0] or faces[1]
                # should_be_outer=True if peer_face==faces[0] else False
                face_turn_wrt_edge=self.face_turn_wrt_edge(peer_face,edge)
                opposite_turn = -face_turn_wrt_edge
                rev_cycle = self.dfs_walk_turning_back(edge, edge[0], opposite_turn)
                assert rev_cycle
                cycle = list(reversed(rev_cycle))
                if not cycle[0][0][0] == cycle[-1][0][1]:
                    assert False
                sum_of_turns=self.sum_of_turns(cycle)
                real_turn = LEFT if sum_of_turns < 0 else RIGHT
                if (face_turn_wrt_edge==real_turn): # outer face
                    # if not should_be_outer:
                    #     assert False
                    self.add_face('outer', [], cycle, opposite_turn) # because outer face
                else: # inner face
                    # if should_be_outer:
                    #     assert False
                    face = self.next_face_name()
                    self.add_face(face, [], cycle, real_turn) # because inner face


######### cutoffs

    def compute_path_cuts(self):
        des = copy.copy(self.dual_edges)
        cutoffs=[]
        adj_faces=dict()
        for path_edge in self.path_edge_dict:
            faces =self.e2face.get(path_edge)
            if not faces:
                assert False
            if faces[0]==None or faces[1]==None:
                assert False

            f0_turn=self.face_turn_wrt_edge(faces[0],path_edge)
            f1_turn=self.face_turn_wrt_edge(faces[1],path_edge)
            f0_turn=f0_turn or -f1_turn
            f1_turn=f1_turn or -f0_turn


            fde=(faces[0],faces[1], path_edge)
            left_face = faces[0]
            right_face = faces[1]
            if f0_turn == RIGHT: # cycle with right turn is on the edge's right
                left_face = faces[1]
                right_face = faces[0]
            adj_faces[(left_face,right_face)]=1

        for left_face,right_face in adj_faces:
            cutoff_list = list(self.dfs_find_all_paths_dual_graph(
                (left_face, right_face, None), left_face, self.max_node_cut
            ))
            for cutoff in cutoff_list:
                cutoffs.append(cutoff)
        self.cutoffs = cutoffs
        return cutoffs

    def enum_cutoffs(self):
        gens=[]
        for dual_paths in self.cutoffs:
            planar_arrays = [ edge[2] for edge in dual_paths ]
            gens.append(itertools.product(*planar_arrays))
        gen_all = itertools.chain(*gens)
        return gen_all

    def enum_node_cutoffs(self):
        return self.node_cutoffs

    def compute_node_cuts(self, max_node_cut=None):
        self.node_cutoffs=[]
        nodesets=dict()
        counter=0
        for cutoff_edges in self.enum_cutoffs():
            for nodes in self.node_cutoff(cutoff_edges):
                if nodesets.setdefault(nodes,counter)==counter:
                    self.node_cutoffs.append(nodes)
                    counter+=1
        return self.node_cutoffs

    def node_cutoff(self,cutoff_edges):
        node_edges=dict()
        for edge in cutoff_edges:
            node_edges.setdefault(edge[0],dict())[edge]=1
            node_edges.setdefault(edge[1],dict())[edge]=1

        node_cuts=[]

        queue=deque([(None,[],cutoff_edges)])
        while queue:
            node,nodes,edges=queue.popleft()
            if node==self.start_node:
                continue
            if node:
                nodes=nodes[:]
                bisect.insort_right(nodes,node)
                edges=[edge for edge in edges if not edge in node_edges[node]]
                if not edges:
                    node_cuts.append(tuple(nodes))
                    continue
            for edge in edges:
                queue.append((edge[0], nodes, edges))
                queue.append((edge[1], nodes, edges))
        return node_cuts
                
    def node_cutoff_(self,cutoff_edges):
        pairs =  [ ((e[0], e[1]), (e[1], e[0])) for e in [self.get_edge(edge) for edge in cutoff_edges] ]
        for vec in itertools.product(*pairs):
            nodes = []
            nodes0 = dict()
            nodes1 = dict()
            for edge in vec:
                if self.start_node==edge[0]:
                    break  # start_node cannot be removed
                nodes0.setdefault(edge[0],dict())[edge[1]]=1

                # if edge[0] in nodes0:
                #     continue
                if edge[0] in nodes1:
                    break # redundant combination : ((1,0),(0,0))... ((0,0),(0,1))
                elif edge[1] in nodes0:
                    continue # edge already broken : ((0,0),(0,1)) ... ((1,0),(0,0))
                nodes0.add(edge[0])
                nodes1.add(edge[1])
                nodes.append(edge[0])
            else:
                yield list(nodes)


    def cutoff_check(self, cutoff_edges, G=None):
        if not self.cutoffs:
            raise("cutoffs not computed")
        TG = G or self.DAG.copy()
        edges = [ self.get_edge(edge) for edge in cutoff_edges ]
        backup_edges=[ (*edge,  G.edges[edge]) for edge in edges ] if G else None
        TG.remove_edges_from(edges)
        path_found=True
        try:
            path =nx.shortest_path(TG,self.start_node,self.end_node)
        except nx.NetworkXNoPath:
            path_found=False
        if backup_edges:
            TG.add_edges_from(backup_edges)
        return not path_found # true == cutoff worked

    

    def cut_probability(self, removals=3):
        cut_count=[0]*removals
        for cutoff in self.enum_node_cutoffs():
            if len(cutoff) > removals:
                continue
            cut_count[len(cutoff)-1]+=1
        prob=0
        prob_rest=1
        combinations=1
        rem_combinations=1
        for i in range(len(cut_count)):
            combinations*=(len(self.DAG.nodes)-i)
            rem_combinations*=(i+1)
            _prob =  cut_count[i]/combinations/rem_combinations
            prob+=_prob
            prob_rest-=_prob
            if not (prob >=0 and prob < 1):
                assert False
            if not prob + prob_rest==1:
                assert False
        return prob

    
    def dfs_find_all_paths_dual_graph(self, start_edge, end_node, max_node_cut):
        edges = self.dual_neighbors_edges(start_edge[0], [ start_edge[1] ] )
        planars = next(edges)[2]
        start_edge=(start_edge[0], start_edge[1], planars)
        queue = deque([(start_edge,dict())])
        tree = dict()
        seen_nodes=dict()
        max_edge_cut= max_node_cut*1 # STUB: appropriate ratio to be determined
        while queue:
            edge,path_dict = queue.pop()
            if max_node_cut and len(path_dict) >= max_node_cut:
                continue
            parent, node, _ = edge
            if node in path_dict:
                continue
            path_dict=copy.copy(path_dict)
            path_dict[node]=1

            # seen_nodes[node] = 1

            if node==end_node:
                path=[]
                while edge:
                    path.insert(0,edge)
                    edge = tree.get(edge[:2])
                yield path
                continue

            edges = self.dual_neighbors_edges(node)
            for nei_edge in edges:
                queue.append((nei_edge,path_dict))
                parent, node, planars = nei_edge
                tree[(parent,node)] = edge
        return


    def dfs_find_path_dual_graph(self, start_edge, end_node):
        edges = self.dual_neighbors_edges(start_edge[0], [ start_edge[1] ] )
        planars = next(edges)[2]
        start_edge=(start_edge[0], start_edge[1], planars)
        queue = deque([start_edge])
        path = []
        tree = dict()
        seen_nodes=dict()
        while queue:
            edge = queue.pop()
            parent, node,_ = edge
            if node in seen_nodes:
                continue
            seen_nodes[node] = 1

            if node==end_node:
                break

            edges = self.dual_neighbors_edges(node)
            for nei_edge in edges:
                queue.append(nei_edge)
                parent, node, planars = nei_edge
                tree[(parent,node)] = edge
        else:
            return path # no cycle found

        while edge:
            path.append(edge)
            edge = tree.get(edge[:2])
        return path

    def dual_neighbors_edges(self,node,tgt_nodes=[]):
        adj_nodes = self.dualG[node]
        for adj_node in adj_nodes:
            if tgt_nodes and not adj_node in tgt_nodes:
                continue
            edge = self.dualG[node][adj_node]
            planars=[]
            for key,value in edge.items():
                planars.append(value['planar_edge'])
            if planars:
                yield (node,adj_node,planars)
        return

    def sum_of_turns(self, arc):
        sum_of_turns=0
        prev_edge=None
        for edge,junction in arc:
            if prev_edge:
                _turns=(turns(prev_edge[2],edge[2])+2)%len(DIRS)-2
                sum_of_turns += _turns
                # sum_of_turns+=(edge[2]-prev_edge[2]+len(DIRS))%len(DIRS)
            prev_edge=edge
        return sum_of_turns


    def assess_path(self,path):
        G=self.G
        for i,node in enumerate(path):
            if i==len(path):
                break
            neis= G[node]
            for nei in neis:
                if nei==path[i+1]:
                    continue

