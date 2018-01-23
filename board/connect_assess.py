from sched import scheduler

import networkx as nx
import random,math,copy
import os, sys
from random import randrange,randint
from collections import deque,defaultdict
from attrdict import AttrDict
from pprint import pprint as pp
import itertools


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

    cycle_sets=AttrDict()
    debug_edge = ((3, 5), (3, 4), 2)
    debug_edges = [debug_edge, back_edge(debug_edge)]
    debug_face = ((4, 4), (3, 4), 3)
    debug_face2 = ((None), (None), 1)

    def __init__(self,G):
        self.G=G
        self.faceG=nx.DiGraph()

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

    def _sp(self,block=False):
        import matplotlib.pyplot as plt

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

        face_starting_edges = [ (face, 'lime' if self.faces[face]['turn']==LEFT else 'r' )
                                for face in self.faces  if face!=self.outer_face]
        nx.draw_networkx_edges(DFG,pos=gna(G,'pos')
                               , edgelist=[ v[0] for v in face_starting_edges ]
                               , edge_color=[ v[1] for v in face_starting_edges ]
                               , width=2.0)
        # de_parents=dict()
        # de_children=dict()
        edge2dual={ de[2]:de for de in self.dual_edges }
        # for de in self.dual_edges:
        #     edge2dual[de[2]]=de
            # de_parents.setdefault(de[1], dict())[de[0]]=de[2]
            # de_children.setdefault(de[0], dict())[de[1]]=de[2]

        de_edges_by_dir = [x[:] for x in [[]] * 4]

        for edge,edge_turn in self.dagp_edges.items():
            face=self.e2face.get(edge)
            # if not self.faces[face]['edges'][edge]:
            #     assert False
            if not face: continue

            de=edge2dual.get(edge)
            rotate = self.faces[face]['turn']

            if face==de[0]: #
                rotate = -self.faces[face]['turn']
            elif face==de[1]:
                rotate = self.faces[face]['turn']
            else:
                assert False
            if (edge_turn==0):
                rotate=-rotate
            de_edges_by_dir[edge[2]].append((edge, rotate))

            # parents = de_parents.get(face,[])
            # children = de_children.get(face,[])
            # incoming=None
            # for linked_faces in [parents, children]:
            #     for link in linked_faces:
            #         if self.faces[link]['edges'].get(edge):
            #             incoming=(linked_faces==parents)
            #     rotate = None
            #     if (incoming==True)!=(edge_turn==0):
            #         rotate = self.faces[face]['turn']
            #     else:
            #         rotate = -self.faces[face]['turn']
            #     de_edges_by_dir[edge[2]].append((edge,rotate))

        # for edge,rotate in de_edges:
        #     de_edges_by_dir[edge[2]].append((edge,rotate))

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
            # print(left_pos)
            _=nx.draw_networkx_edge_labels(DLG,pos=left_pos , label_pos=0.5, edge_labels=left_labels, font_size=12, font_color='lime')
            _=nx.draw_networkx_edge_labels(DRG,pos=right_pos , label_pos=0.5, edge_labels=right_labels, font_size=12, font_color='r')


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


    def cycles_along_path(self, start_path):
        arc_junctions=self.arc_junctions=dict()
        start =start_path[0]
        path_dict = self.path_dict = self.get_nodes_dict(start_path)
        dagp=self.dagp = defaultdict(list)
        dagp_rev=self.dagp_rev=defaultdict(list)
        dagp_edges=self.dagp_edges=dict()
        trunk_join=self.trunk_join=dict()
        order = self.order=dict()
        self.reachable_parents=dict()
        self.reachable_children=dict()
        counter=self.counter=0
        e2face=self.e2face=dict()
        faces=self.faces=dict()
        dual_edges=self.dual_edges=dict()
        outer_cycle= self.outer_cycle=[]
        outer_edges= self.outer_edges=dict()

        trunk_turn =0

        dagp[start_path[-1]]=[]
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
                    nonlocal self
                    edge, suggested_turn = item
                    tup, next_arc, arc_back, sum_of_turns, cycle_found, *_ = [None] * 10
                    for edge_turn in [suggested_turn, -suggested_turn]:
                        outer_turn = self.outer_edges.get(edge)
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
                        if not sum_of_turns:
                            assert False
                        if (sum_of_turns < 0 and edge_turn == LEFT
                                or sum_of_turns > 0 and edge_turn == RIGHT):
                            self.outer_cycle = _next_arc + arc_back
                            for outer_edge, junction in self.outer_cycle:
                                self.outer_edges[outer_edge] = edge_turn
                                self.outer_edges[back_edge(outer_edge)] = -edge_turn
                            continue
                        next_arc = _next_arc
                        break  # normal arc

                    if not next_arc:
                        if cycle_found:
                            scheduled_late.append((edge, suggested_turn))
                        return

                    face_turn = LEFT if sum_of_turns > 0 else RIGHT
                    # face_turn_wrt_start_edge=None

                    arc_face = None
                    for arc_edge, junction in next_arc:
                        self.add_to_dag(arc_edge, edge_turn)
                        if not arc_face:
                            arc_face = arc_edge # face labeled after the edge that formed it
                            faces[arc_face]={'edges': {arc_edge:1}, 'turn': face_turn}
                        else:
                            faces[arc_face]['edges'][arc_edge] = 1
                        e2face[arc_edge] = arc_face

                    peer_faces = dict()
                    for arc_edge, junction in arc_back:
                        fw_face = e2face.get(arc_edge)
                        bw_face = e2face.get(back_edge(arc_edge))
                        if bool(fw_face) and bool(bw_face):
                            assert False
                        if not fw_face and not bw_face:
                            for path_edge in (arc_edge, back_edge(arc_edge)):
                                if self.dagp_edges.get(path_edge)==0: # path edge
                                    faces[arc_face]['edges'][path_edge] = 1
                                    e2face[path_edge] = arc_face
                            continue
                        de_edge=None
                        if fw_face:
                            # peer_faces[fw_face] = 'from' if face_turn == RIGHT else 'to'
                            de_edge = (fw_face, arc_face, arc_edge) if face_turn==RIGHT else (arc_face, fw_face, arc_edge)
                            faces[arc_face]['edges'][arc_edge] = 1
                        elif bw_face:
                            # peer_faces[bw_face] = 'from' if face_turn == LEFT else 'to'
                            de_edge = (bw_face, arc_face, back_edge(arc_edge)) if face_turn == LEFT else (arc_face, bw_face, back_edge(arc_edge))
                            faces[arc_face]['edges'][back_edge(arc_edge)] = 1

                        assert de_edge
                        dual_edges[de_edge] = 1


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

        for item_i in range(len(scheduled_late)):
            process_item(scheduled_late[item_i])

        def process_outer_cycle():
            if not self.outer_cycle:
                assert False # STUB
            outer_node=((None,None))
            outer_face = self.outer_face=(outer_node,outer_node)
            for arc_edge,junction in self.outer_cycle:
                de_edge=None
                for peer_edge in (arc_edge,back_edge(arc_edge)):
                    peer_face = e2face.get(peer_edge)
                    if not peer_face:
                        continue
                    face_turn = self.faces[peer_face]['turn']
                    if self.dagp_edges.get(peer_edge)==0:
                        face_turn=-face_turn
                    de_edge = (outer_face, peer_face, peer_edge) if face_turn == RIGHT else (peer_face, outer_face, peer_edge)
                    self.faces.setdefault(outer_face, {'edges': dict(), 'turn': LEFT})['edges'][peer_edge] = 1
                    break
                assert de_edge
                dual_edges[de_edge] = 1
        process_outer_cycle()

        def check_dual_graph():
            pass


        return dagp_edges


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

    def dfs_walk(self, start_edge, turn):
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

    def dfs_walk_back(self, start_edge, end_node, turn):
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
                if edge!=start_edge:
                    arc_map[nei_edge] = edge

        while edge:
            arc.append((edge, None))
            edge = arc_map.get(edge)
        return arc

    def arc_walk(self, start_edge, turn):
        arc  = self.dfs_walk(start_edge,turn)
        if not arc:
            return 'noose'  #  dead end
        if arc[0][0][1]==arc[-1][0][0]:
            return 'cycle'  #  cycle


        reachable = self.reachable_parents.get(arc[-1][0][0]) # BETTER: classes instead of tuples
        if reachable and reachable.get(arc[0][0][1]):
            return 'cycle_part' # forms a cycle

        arc_back = self.dfs_walk_back(arc[0][0], start_edge[0], turn)
        assert bool(arc_back)

        sum_of_turns=0
        prev_edge=None
        for edge,junction in itertools.chain(arc_back,arc):
            if prev_edge:
                _turns=(turns(prev_edge[2],edge[2])+2)%len(DIRS)-2
                sum_of_turns += _turns
                # sum_of_turns+=(edge[2]-prev_edge[2]+len(DIRS))%len(DIRS)
            prev_edge=edge
        return (list(reversed(arc)), list(reversed(arc_back)), sum_of_turns)

    def assess_path(self,path):
        G=self.G
        for i,node in enumerate(path):
            if i==len(path):
                break
            neis= G[node]
            for nei in neis:
                if nei==path[i+1]:
                    continue

