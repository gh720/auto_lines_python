import copy
import random
from random import randrange,randint
import collections
import bisect
from sortedcontainers import SortedListWithKey, SortedDict
import pickle
import base64
import itertools
import math
from typing import List, Set, Dict, Tuple, Text, Optional, AnyStr

from .utils import ddot,dpos,ddir
from .board_graph import Board_graph
from .connect_assess import connect_assessor_c


'''
what can be done:
1. placing at adj free cells
2. -- (4, 5) to (4, 3): 0.000000 - should not be (max_edge_cut not big enough ?)
3. should have assessed (0,5 - 4,5) candidate first
4. moving obstacles
4.1 checking if moved obstacle can be a candidate
5. checking overall connectivity
6. priority among color sharing candidates
7. check path existence during assessment
8. assess cand's free cells connectivity
9. consider available colors
10. filling
10.1 other balls won't be blocked
10.2 other balls aren't blocked

tentative graph use cases:
1. assess cut probability
2. check connectivity in case node removed
2. 


'''

class Board:
    # COLORS=['Y','B','P','G','C','R','M']
    _colors="red green yellow blue purple cyan magenta".split(' ')
    _colsize=len(_colors)

    _dirs = [
        [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]
    ]

    NOT_A_CELL=dpos(-1,-1)

    _straight_dirs=[]
    for dir in _dirs:
        if dir[0] and dir[1]:
            continue
        _straight_dirs.append(ddir(dir[0],dir[1]))

    _color_list = None
    _array: List[List[str]] = None
    _size = None
    _scrub_length=None
    _scrubs=None
    _tentative_scrubs=None
    _bg:Board_graph=None
    _assessment=None
    _axes= None

    iteration = None
    current_move = None
    fake_prob=True

    _history=list()

    debug_moves=[
        ((1,8),(0,4)),
        # ((1, 8), (0, 4))
    ]
    # axes: x - left, y - up, directions enumerated anti-clockwise

    def __init__(self, size=9, batch=5, colsize=None, scrub_length=5, axes=None, logfile=None):
        if colsize==None:
            colsize=len(self._colors)
        self._size=size
        self._batch=batch
        self._scrub_length=scrub_length
        self._colsize=colsize
        self._sides=[[0,None], [None,0], [self._size-1,None], [None,self._size-1]]
        self._color_list=collections.defaultdict(dict)
        self._axes = axes
        # self.update_buffer()
        self.reset()
        self._bg.update_graph(self)
        # self.prepare_view()
        self.max_color_candidates=10
        self.max_free_moves=100
        self.max_obst_moves=100
        self.max_no_path_moves=100
        self.components=list()
        self._logfile=logfile
        if self._logfile:
            wh = open(self._logfile, 'w')
            wh.close()


    def reset(self):
        # COLORS=['Y','B','P','G','C','R','M']
        self._array = [
            [None for i in range(0, self._size)]
            for j in range(0, self._size)
        ]
        self._color_list = dict()
        self._scrubs = []
        self._bg = Board_graph(self._axes)
        self.iteration = 0
        self.current_move = None
        self.reset_assessment()


    def get_array(self):
        return self._array

    def get_size(self):
        return self._size

    def log(self, msg):
        if self._logfile:
            with open(self._logfile, 'a') as wh:
                wh.write(msg+"\n")
        print(msg)

    def reset_assessment(self):



        def debug_sort(color):
            r = color.move_in_out

        cand_color_sort = lambda color: color.move_in_out
        self._assessment=ddot(moves=list()
                              , move_map=dict()
                              , candidates=list()
                              , cand_colors=SortedListWithKey(key=cand_color_sort)
                              , cand_color_moves=SortedListWithKey(key=lambda move: -move.total_gain)
                              , cand_color_map=dict()
                              , cand_map=dict()
                              , cand_free_map=dict()
                              , no_path_map = dict()
                              , no_path_moves=SortedListWithKey(key=lambda move: -move.total_gain)
                              )
            # ddot(
            # move_map=dict()
            # , candidates=list()
            # , move_in=None, move_out=None, move_in_out=None
        # )

    def get_cell(self,cell:dpos) -> str:
        return self._array[cell.x][cell.y]

    def set_cell(self,x,y,value):
        self._array[x][y]=value

    # line defined by axes crossing: a: for y=0, b for x=0, if None then it is parallel to that axis
    def ray_hit(self, x,y,dx,dy,line):
        a,b=line
        if a==None: # horizontal
            if dy==0: # parallel
                return None
            if dx==0:
                return [x,b]
            return [x + dx*dy*(b-y),b]
        if b==None: # vertical
            if dx==0: # parallel
                return None
            if dy==0:
                return [a,y]
            return [a, y + dx*dy*(a-x)]

    # def sign(x):
    #     if x==0:
    #         return 0
    #     return -1 if x<0 else 1
    #
    # def get_segment(self,x,y,dx,dy):
    #     line=None
    #     length=None
    #     for side in self._sides:
    #         hit = self.ray_hit(x,y,dx,dy,side)
    #         if hit==None:
    #             continue
    #         x0,y0=hit
    #         if dx!=sign(x0-x):
    #             continue
    #         if dy!=sign(y0-y):
    #             continue
    #         _mx = max(abs(y0-y), abs(x0-x))
    #         if length==None or _mx < length:
    #             line=[x,y,dir_index, length ]
    #     assert line!=None
    #     return line
    #
    # def get_line(self, x,y, dx,dy):
    #     line = self.get_segment(x,y,dx,dy)
    #     line.append(self.get_segment_items(x,y,dx,dy,length))
    #     return line


    # returns x,y and length of a board's chord for given dx,dy
    def get_starts(self,dx,dy):
        starts=None
        if dx==0:
            starts=[(x,0 if dy>0 else self._size-1,self._size) for x in range(0,self._size)]
        elif dy==0:
            starts=[(0 if dx>0 else self._size-1, y, self._size) for y in range(0,self._size)]
        elif dx==dy:
            count = max(0,self._size-self._scrub_length+1)
            starts=([ (0, y, self._size-y) if dx>0 else (self._size-1, self._size-y-1, self._size-y) # vert for /
                        for y in range(0,count) ] + 
                    [ (x, 0, self._size-x) if dx>0 else (self._size-x-1, self._size-1, self._size-x) # horiz for /
                        for x in range(1,count) ] )
        elif dx==-dy:
            count = max(0,self._size-self._scrub_length+1)
            starts=([ (0, self._size-y-1, self._size-y) if dx>0 else (self._size-1, y, self._size-y) # vert for \
                        for y in range(0,count) ] + 
                    [ (x, self._size-1, self._size-x) if dx>0 else (self._size-x-1, 0, self._size-x) # horiz for \
                        for x in range(1,count) ])
        assert starts!=None
        return starts


    def get_segment_items(self,start_cell:dpos,dir:ddir,length=None) -> List[Tuple[dpos,str]]:
        items=[]
        cell = start_cell.copy()
        # cx:int=cell.x
        # cy:int=cell.y
        i=0
        while self.valid_cell(cell):
            if length!=None and i >=length:
                break
            items.append([cell.copy(),self.get_cell(cell)])
            cell.x+=dir.dx
            cell.y+=dir.dy
            i+=1

        return items

    def get_chords(self):
        lines=[]
        for k in range(0,4):
            dx,dy=self._dirs[k]
            starts=self.get_starts(dx,dy)
            for x,y,length in starts:
                items=self.get_segment_items(dpos(x,y),ddir(dx,dy),length)
                lines.append((dpos(x,y),ddir(dx,dy),length,items))
        return lines


    def candidate(self,_cell:dpos,_dir:ddir,items):
        cell=_cell.copy()
        dir= _dir.copy()
        A=self._assessment
        cand = ddot(colors=dict(),free=dict(),cells=dict(), ball_count=0, move_in_out=0
                    , index=None, rank=None, moves=dict(), start=cell, dir=dir)
        # colors = dict()
        # colors = AttrDict(color=AttrDict(pos=list(),count=0))
        # free=dict()
        # ball_count=0
        for i in range(len(items)):
            item_cell,color_name=items[i]
            cand.cells[item_cell]=color_name
            if color_name!=None:
                color= cand.colors.setdefault(color_name, ddot(cells=list(),name=color_name,count=0,cand=cand
                        , lc = None
                        , moves=SortedListWithKey(key=lambda move: (
                            -move.gain
                            ))
                   ))

                color.cells.append(ddot(i=i, cell=item_cell))
                color.count+=1
                cand.ball_count+=1
            else:
                cand.free[item_cell]={'lc':self._bg.metrics['lc'][tuple(item_cell)]}

        cand_colors = []
        for color_name, color in cand.colors.items():
            color.move_in_out = len(cand.free) + (cand.ball_count - color.count)*2
            cand_colors.append(color)
            for item in color.cells:
                A.cand_color_map.setdefault(item.cell,[]).append(color)
            for item in cand.cells:
                A.cand_map.setdefault(item, []).append(color)


        for pos, free in cand.free.items():
            A.cand_free_map.setdefault(pos,[]).append(cand)

        return cand,cand_colors


    def candidate_iter(self):
        A = self._assessment
        lines = self.get_chords()
        for line in lines:
            cell, dir, length, items = line
            for i in range(0, length - self._scrub_length + 1):
                cand_key = (cell,dir)
                if cand_key in A.cand_map:
                    continue
                cand, colors = self.candidate(cell, dir, items[i:i + self._scrub_length])
                # cand.index=len(self._cand_array)
                A.candidates.append(cand)
                for color in colors:
                    A.cand_colors.add(color)
                yield cand

    def candidates(self):
        A=self._assessment
        lines = self.get_chords()
        best_cand=None
        for line in lines:
            cell,dir,length,items=line
            for i in range(0, length-self._scrub_length+1):
                cand,colors = self.candidate(items[i][0],dir,items[i:i+self._scrub_length])
                # cand.index=len(self._cand_array)
                A.candidates.append(cand)
                for color in colors:
                    A.cand_colors.add(color)

    def find_best_move(self):
        if self.iteration==2:
            debug=1
        self.reset_assessment()
        self.candidates()
        # move =
        # while True:
        #
        #     for cand_color in self._assessment.cand_colors:
        #     if not cand:
        #         cand = self.candidate_iter()
        #         if not cand:
        #             break
        #     for color in cand.colors:
        #
        #     A.candidates.append(cand)
        #     for color in colors:
        #         A.cand_colors.add(color)
        #

        self.color_assessment()
        best_move = self.pick_best_move()
        self.check_tentative_scrubs(best_move.pos_from,best_move.pos_to,best_move.color.name)
        return best_move

    def pick_best_move(self):
        A=self._assessment
        best_move=None
        for move in A.cand_color_moves:
            best_move = move
            break
        # for cand_color in A.cand_colors:
        #     for move in cand_color.moves:
        #         best_move= move
        #         break
        #     else:
        #         continue
        #     break
        # else:
        #     return None

        return best_move


    def color_assessment(self):
        counter=0
        A=self._assessment
        for cand_color in A.cand_colors:
            if self.assess_color_placement(cand_color):
                counter+=1
                if counter>self.max_color_candidates:
                    break

        def no_path_loop():
            counter = 0
            cross_dict = dict()
            for no_path_move in A.no_path_moves:
                cross = no_path_move.cross
                if not cross:
                    continue
                for node in cross:
                    cross_dict.setdefault(node, SortedListWithKey(key=lambda move: -move.total_gain)).add(no_path_move)
            if not cross_dict:
                return
            nodes = sorted(cross_dict.keys(), key=lambda node: -cross_dict[node][0].total_gain)

            for node in nodes:
                blocked_moves = cross_dict[node]
                for blocked_move in blocked_moves:
                    if blocked_move.color.name == self.get_cell(dpos(node[0],node[1])): # should not happen - taken care of earlier
                        continue

                    block_color_cands = A.cand_color_map.get(dpos(node[0],node[1]),[])
                    for block_color in block_color_cands:
                        if block_color.name == blocked_move.color.name: # useless to move away an obstacle of the same color
                            continue
                        color_cells = self.get_cells_by_color(block_color.name)
                        for pos in color_cells:
                            if pos == node:
                                continue
                            dst_colors_cands = A.cand_color_map[pos]
                            for dst_color in dst_colors_cands:
                                if dst_color.name != block_color.name:
                                    continue
                                for free_pos, metric in dst_color.cand.free.items():
                                    for pos_item in block_color.cells:
                                        if self.debug_moves:
                                            if (tuple(pos_item.cell), tuple(free_pos)) not in self.debug_moves:
                                                debug = 1
                                                continue
                                            else:
                                                debug = 1

                                        move = ddot(pos_from=pos_item.cell, pos_to=free_pos, color=block_color)
                                        if self.add_move_candidate(move, added_gain = cross_dict[node][0].total_gain, blocked=False):
                                            counter += 1
                                            if counter > self.max_no_path_moves:
                                                return
                    break # clearing the path for only the top blocked move

        no_path_loop()


    def assess_color_placement(self, cand_color) -> None:

        A=self._assessment
        added = False
        def free_loop():
            nonlocal A
            counter = 0
            if cand_color.cand.free:
                for free_pos, metric in cand_color.cand.free.items():
                    color_cells = self.get_cells_by_color(cand_color.name)
                    for pos in color_cells:
                        if pos in cand_color.cand.cells:
                            continue
                        if self.debug_moves:
                            if (tuple(pos), tuple(free_pos)) not in self.debug_moves:
                                debug=1
                                continue
                            else:
                                debug=1
                        move = ddot(pos_from=pos, pos_to=free_pos, color=cand_color)
                        if self.add_move_candidate(move):
                            added = True
                            counter+=1
                            if counter>self.max_free_moves:
                                return

        free_loop()

        def obst_loop():
            nonlocal A
            counter = 0
            for ob_color_name,ob_color in cand_color.cand.colors.items():
                if ob_color.name==cand_color.name:
                    continue
                color_cells = self.get_cells_by_color(ob_color.name)
                for pos in color_cells:
                    if pos in cand_color.cand.cells:
                        continue
                    dst_colors_cands = A.cand_color_map[pos]
                    for dst_color in dst_colors_cands:
                        if dst_color.name!=ob_color.name:
                            continue
                        for free_pos, metric in dst_color.cand.free.items():
                            for pos_item in ob_color.cells:
                                if self.debug_moves:
                                    if (tuple(pos_item.cell), tuple(free_pos)) not in self.debug_moves:
                                        debug=1
                                        continue
                                    else:
                                        debug=1
                                move = ddot(pos_from=pos_item.cell, pos_to=free_pos, color=ob_color)
                                if self.add_move_candidate(move):
                                    added = True
                                    counter+=1
                                    if counter>self.max_obst_moves:
                                        return
        obst_loop()
        return added

    def color_cand_key(self,color):
        return (color.name, color.cand.start, color.cand.dir)

    def cost(self, move:ddot):
        A=self._assessment

        src_cand_colors = A.cand_map[move.pos_from]
        tgt_cand_colors = A.cand_map[move.pos_to]

        src_dict=dict()
        tgt_dict=dict()
        shared=dict()

        for src in src_cand_colors:
            if src.name!=move.color.name:
                continue
            src_key = self.color_cand_key(src)
            for tgt in tgt_cand_colors:
                if tgt.name != src.name:
                    continue
                tgt_key = self.color_cand_key(tgt)
                sh = shared[src_key,tgt_key]=set([v.cell for v in src.cells ]) & set([v.cell for v in tgt.cells ])



        ### tgt cost

        mx_tgt_gain=0
        mx_tgt_loss=0
        color_mio = [None] * 8
        for tgt in tgt_cand_colors:
            if tgt.name != move.color.name:
                if move.pos_from in tgt.cand.cells:
                    loss=0
                else:
                    loss = self._scrub_length - (tgt.move_in_out + 2)
                mx_tgt_loss = max(loss, mx_tgt_loss)
            else:
                if move.pos_from in tgt.cand.cells:
                    color_mio[tgt.move_in_out] = 0
                else:
                    color_mio[tgt.move_in_out] = max(
                        (color_mio[tgt.move_in_out] or 0)
                        , self._scrub_length + 1 - tgt.move_in_out
                    )

        for gain in color_mio:
            if gain != None:
                mx_tgt_gain = gain
                break


        mx_tgt_gain = 100 if mx_tgt_gain == self._scrub_length else max(mx_tgt_gain, 0)
        mx_tgt_loss = max(mx_tgt_loss, 0)

        ### src cost

        mx_src_gain=0
        mx_src_loss=0

        for src in src_cand_colors:
            if src.name == move.color.name:
                if move.pos_to in src.cand.cells:
                    loss = 0
                else:
                    src_key=self.color_cand_key(src)
                    shared_tgt=shared.get(src_key,dict())
                    sh_mx=0
                    for tgt in shared_tgt:
                        tgt_key=self.color_cand_key(tgt)
                        sh_mx=max(sh_mx, len(shared_tgt[tgt_key]))
                    loss=5-src.move_in_out-sh_mx
                mx_src_loss=max(loss,mx_src_loss)
            else:
                if move.pos_to in src.cand.cells:
                    gain=0
                else:
                    gain = 5-src.move_in_out
                mx_src_gain=max(gain,mx_src_gain)

        mx_src_gain = max(mx_src_gain,0)
        mx_src_loss = max(mx_src_loss,0)

        return mx_src_gain, mx_src_loss, mx_tgt_gain, mx_tgt_loss

    def mio_tgt_gain(self, move:ddot):
        A=self._assessment

        pos = move.pos_to
        src_color=move.color.name
        tgt_color = self.get_cell(pos)
        assert tgt_color==None
        tgt_cands=A.cand_free_map[pos]
        impr_mio=None
        detr_mio=None

        mx_gain=-math.inf
        mx_loss=-math.inf
        color_mio=[None]*8
        for cand in tgt_cands:
            for color_name,color in cand.colors.items():
                if color_name == src_color:
                    if move.pos_from in color.cand.cells:
                        color_mio[color.move_in_out] = 0
                        # gain=0
                    else:
                        color_mio[color.move_in_out] = max(
                            (color_mio[color.move_in_out]or 0)
                            , self._scrub_length+1-color.move_in_out
                        )
                        # gain=5-color.move_in_out
                    # if impr_mio == None or impr_mio > color.move_in_out:
                    #     impr_mio = color.move_in_out
                    # mx_gain=max(gain,mx_gain)
                else:
                    if move.pos_from in color.cand.cells:
                        loss=0
                    else:
                        loss=self._scrub_length-(color.move_in_out+2)
                    mx_loss = max(loss, mx_loss)
                    # if detr_mio == None or detr_mio > color.move_in_out:
                    #     detr_mio = color.move_in_out

        for gain in color_mio:
            if gain != None:
                mx_gain = gain
                break

        mx_gain = 100 if mx_gain==self._scrub_length else max(mx_gain,0)
        mx_loss = max(mx_loss,0)

        # gain = 0 if impr_mio==None else (100 if impr_mio<=1 else self._scrub_length - impr_mio)
        # loss = 0 if detr_mio==None else (100 if detr_mio<=1 else self._scrub_length - detr_mio)

        return mx_gain, mx_loss

    def mio_src_gain(self, move:ddot):
        A=self._assessment
        pos=move.pos_from
        src_color=move.color.name
        assert self.get_cell(dpos(pos.x,pos.y)) == src_color

        color_cands = A.cand_map[pos]

        mx_gain=-math.inf
        mx_loss=-math.inf
        for color in color_cands:
            if color.name == src_color:
                if move.pos_to in color.cand.cells:
                    loss=0
                else:
                    loss=5-color.move_in_out
                # if detr_mio == None or detr_mio > color.move_in_out:
                #     detr_mio = color.move_in_out
                mx_loss=max(loss,mx_loss)
            else:
                if move.pos_to in color.cand.cells:
                    gain=0
                else:
                    gain = 5-color.move_in_out
                    # if impr_mio == None or impr_mio > color.move_in_out:
                    # impr_mio = color.move_in_out
                mx_gain=max(gain,mx_gain)

        mx_gain = max(mx_gain,0)
        mx_loss = max(mx_loss,0)

        # gain = 0 if impr_mio == None else self._scrub_length - impr_mio
        # loss = 0 if detr_mio == None else self._scrub_length - detr_mio

        return mx_gain, mx_loss


    def obst_block_check(self, pos, cand_color) ->float:
        A = self._assessment

        cost_max=0
        for color_name,color  in cand_color.cand.colors.items():
            if color_name==cand_color.name:
                continue
            for cell in color.cells:
                lc_before = self.tent_lc_metric(cell.cell)
                self._bg.change_free_graph(self, free_cells=[],fill_cells=[pos])
                lc_after = self.tent_lc_metric(cell.cell)
                self._bg.change_free_graph(self, free_cells=[pos],fill_cells=[])
                cost=math.inf
                if lc_after==0:
                    if lc_before==0:
                        cost=0
                    else:
                        cost=math.inf
                elif lc_before<=lc_after:
                    cost=0
                else:
                    cost = 1-lc_after/lc_before
                if cost>self._scrub_length:
                    cost=self._scrub_length
                cost_max=max(cost,cost_max)
        return cost_max

    def move_key(self,move:ddot):
        move_key = (tuple(move.pos_from), tuple(move.pos_to))
        return move_key

    def cand_cost(self,move:ddot):
        cand = move.color.cand
        mx_gain = 0
        mx_loss = 0
        for cname, color in cand.colors.items():
            if cname!=move.color.name:
                for cell in color.cells:
                    gain,loss = self.mio_src_gain(ddot(pos_from=cell.cell, pos_to=self.NOT_A_CELL, color=color))
                    mx_gain = max(gain, mx_gain)
                    mx_loss = min(loss, mx_loss)
        return - mx_loss



    def add_move_candidate(self, move:ddot, added_gain:float=0, blocked=None):
        """
        :returns: bool - True if move was added, False if not (path blocked)
        """
        cand_color=move.color
        A=self._assessment
        move_key= self.move_key(move)
        if move_key in A.move_map:
            return

        if move_key in A.no_path_map:
            return

        gd = ddot(gain=0, loss=0, cand_cost=0, added_gain=0, src_gain=0, src_loss=0
                               , tgt_gain=0, tgt_loss=0, ob_block_cost=0
                               , lc=0, cut_prob=0)

        gd.added_gain = added_gain
        # gd.src_gain, gd.src_loss = self.mio_src_gain(move)
        # gd.tgt_gain, gd.tgt_loss = self.mio_tgt_gain(move)
        gd.src_gain, gd.src_loss,  gd.tgt_gain, gd.tgt_loss = self.cost(move)

        gd.gain = max(gd.src_gain, gd.tgt_gain, gd.added_gain)
        gd.loss = max(gd.src_loss, gd.tgt_loss)

        gd.cand_cost = self.cand_cost(move)

        gd.ob_block_cost = self.obst_block_check(move.pos_to, cand_color)

        path_exists, cross = self.check_path_across(move.pos_from, move.pos_to)

        if not path_exists:
            A.no_path_map[move_key] = move
            if blocked!=False:
                move.total_gain = (gd.gain - gd.loss + gd.cand_cost) * 1.0 - gd.ob_block_cost - gd.lc * 2 + gd.cut_prob
                move.gain_detail=gd
                trimmed_cross = set()
                for node in cross:
                    if self.get_cell(dpos(node[0], node[1]))!=move.color.name:
                        trimmed_cross.add(node)

                move.cross = trimmed_cross
                A.no_path_moves.add(move)
                self.log("%s %d,%d>%d,%d: %.2f: g,l,cc,ag=%.2f,%.2f,%.2f,%.2f ob=%.2f cross=%s"
                         % ('-', move.pos_from.x, move.pos_from.y, move.pos_to.x, move.pos_to.y
                            , move.total_gain, gd.gain, gd.loss, gd.cand_cost, gd.added_gain
                            , gd.ob_block_cost, cross))
                return True
            return False

        gd.cut_prob = self.assess_cut_probability(move)
        assert gd.cut_prob!=None
        gd.lc = self.tent_lc_metric(move.pos_from, move.pos_to)

        move.total_gain = (gd.gain - gd.loss + gd.cand_cost) * 1.0 - gd.ob_block_cost - gd.lc * 2 + gd.cut_prob
        move.gain_detail=gd

        A.move_map[move_key] = move
        A.cand_color_moves.add(move)

        self.log("%s %d,%d>%d,%d: %.2f: g,l,cc,ag=%.2f,%.2f,%.2f,%.2f ob,lc,cp=%.2f,%.2f,%.2f"
                 % ('+', move.pos_from.x, move.pos_from.y, move.pos_to.x, move.pos_to.y
                    , move.total_gain, gd.gain, gd.loss, gd.cand_cost, gd.added_gain
                    , gd.ob_block_cost, gd.lc, gd.cut_prob))
        return True

    def tent_lc_metric(self, cell:dpos, end_cell:dpos=None) -> float:
        self._bg.change_free_graph(self, free_cells=[cell], fill_cells=[])
        lc:float= None
        path_ok=True
        if end_cell:
            if not self._bg.check_path(tuple(cell), tuple(end_cell)):
                path_ok=False
        if path_ok:
            lc = self._bg.metric('lc', (cell.x,cell.y))
        self._bg.change_free_graph(self, free_cells=[], fill_cells=[cell])

        return lc

    def cell_components(self, cell):
        return self.component_map.get(tuple(cell))

    def check_path_across(self, cell_from, cell_to):
        comps1 = self.cell_components(cell_from)
        if not comps1:
            return False, None

        comps2 = self.cell_components(cell_to)
        assert len(comps2)==1
        comp2 = comps2[0]

        cross =set()
        path_exists=False
        if self.get_cell(cell_from) != None:
            for comp1 in comps1:
                if comp1[0] == comp2[0]:
                    path_exists = True
                else:
                    cross |= comp1[1] & comp2[1]
        else:
            assert len(comps1)==1
            for comp1 in comps1:
                if comp1[0] == comp2[0]:
                    path_exists = True
                else:
                    cross |= comp1[1] & comp2[1]

        return path_exists, cross

    # might be not necessary
    def assess_cand_placement(self, cand):

        # A=self._assessment
        if cand.free:
            scolors=sorted(cand.colors.items(), key=lambda i:i[1].count, reverse=True)
            for i_color, color in scolors:
                for free_pos, metric in cand.free.items():
                    color_cells = self.get_cells_by_color(i_color)
                    for pos in color_cells:
                        if pos in cand.cells:
                            continue

                        has_path = self.check_path(pos,free_pos)
                        move = ddot(pos_from=pos, pos_to=free_pos, color=color
                                    , gain=None, gain_detail=None, cross=None)
                        self.add_move_candidate(move)

    def assess_cut_probability(self, move:ddot):
        # adj_cells= self.free_adj_cells(start_pos)
        # for cell in adj_cells:
        #     start_node = cell.x

        start_node=(move.pos_from.x, move.pos_from.y)
        end_node=(move.pos_to.x, move.pos_to.y)
        adj_free = self.free_adj_cells(move.pos_from)
        start_edges= [ (start_node, tuple(adj) ) for adj in adj_free]
        if not self.fake_prob:
            ca = self._bg.assess_connection_wo_node(start_node, end_node, start_edges, max_cut=3)
            if ca==None:
                cut_prob=None
            else:
                cut_prob = ca.cut_probability()
            return cut_prob
        else:
            self._bg.change_free_graph(self, free_cells=[move.pos_from], fill_cells=[])
            lc:float=None
            if self._bg.check_path(start_node, end_node):
                lc = self._bg.fake_assess_connection_wo_node(start_node, end_node, max_cut=3)
            self._bg.change_free_graph(self, free_cells=[], fill_cells=[move.pos_from])

            if lc==None:
                return None
            cut_prob=1-lc
            # cut_prob=0
            # if lc==None:
            #     print("%s to %s: none" % (start_node, end_node))
            #     return None
            # if lc == 0:
            #     cut_prob==1000
            # else:
            #     cut_prob = 1/lc
            # print ("%s to %s: %f" % (start_node, end_node, cut_prob))
        return cut_prob

    def adj_cell(self, cell:dpos, dir:ddir):
        d=dpos(cell.x+dir.dx, cell.y+dir.dy)
        if self.valid_cell(d):
            return d
        return None

    def check_pos(self, pos:dpos):
        if pos.x<0 or pos.y<0 or pos.x>=self._size or pos.y>=self._size:
            return False
        return True

    def free_adj_cells(self, start_cell):
        cells=[]
        for dir in self._straight_dirs:
            cell:dpos = self.adj_cell(start_cell, dir)
            if not cell:
                continue
            color = self.cell(cell)
            if color:
                continue
            cells.append(cell)
        return cells

    def adj_cells(self, start_cell):
        cells=[]
        for dir in self._straight_dirs:
            npos:dpos = self.new_pos(start_cell, dir)
            if not npos:
                continue
            color = self.cell(npos)
            if color:
                continue
            cells.append(npos)
        return cells

    def next_move(self):
        history_item = dict(board=dict(move=list(), remove=list(), new=list()), random_state=None)
        scrubbed=False
        if self.current_move:
            history_item['board']['move'].append((self.current_move.pos_from
                                        , self.current_move.pos_to
                                        , self.current_move.color.name
                                        , self.iteration
                                        , self.current_move.total_gain
                                        , self.current_move.gain_detail
                                                  ))
            self.make_move(self.current_move)
            history_item['board']['remove']=self._scrubs
            if self._scrubs:
                self.scrub_cells()
                scrubbed=True

        self.picked=[]
        if not scrubbed:
            self.picked= self.get_random_free_cells()
            self.place(self.picked)
        history_item['board']['new']=self.picked
        history_item['random_state'] = base64.b64encode(pickle.dumps(random.getstate())).decode('ascii')

        if len(self._history) > self.iteration:
            self._history=self._history[:self.iteration]
        assert len(self._history) == self.iteration
        self._history.append(history_item)
        self.iteration+=1
        self.log("iteration: %d" % (self.iteration))
        # self._bg.update_graph(self)
        self.update_graph()
        self.current_move = self.find_best_move()
        return self.picked

    def history_post_load(self):
        print("iteration: %d" % (self.iteration))
        self.update_graph()
        # self._bg.update_graph(self)
        self.current_move = self.find_best_move()
        return self.picked

    def draw(self, show=False):
        self._bg.init_drawing(self._axes)
        self._bg.draw()
        if show:
            self._bg.show()

    def draw_move(self):
        self._bg.init_drawing(self._axes)
        self._bg.draw()
        start_edges=None
        tentative_scrub_edges=None
        if not self.current_move:
            f=t=None
        else:
            move=self.current_move
            f=move.pos_from
            t=move.pos_to
            start_node = (move.pos_from.x, move.pos_from.y)
            end_node = (move.pos_to.x, move.pos_to.y)
            adj_free = self.free_adj_cells(move.pos_from)
            start_edges = [(start_node, tuple(adj)) for adj in adj_free]
            if self._tentative_scrubs:
                ts= self._tentative_scrubs
                tentative_scrub_edges = [((ts[i].x, ts[i].y),(ts[i+1].x, ts[i+1].y)) for i in range(len(ts)-1)]
        self._bg.draw_move(self, f,t, start_edges, tentative_scrub_edges)
        self._bg.show()

    def get_cells_by_color(self,color):
        return self._color_list[color]

    def place(self,picked):
        for item in picked:
            (pos,color)=item
            if self._scrubs:
                self.scrub_cells()
            self.fill_cell(pos,color)
            self._scrubs = self.check_scrubs(pos)
        self.update_graph()
        # self.component_map = self._bg.get_components()
        # self.update_buffer()

    def place_history(self,picked):
        for item in picked:
            (pos,color)=item
            self.fill_cell(pos,color)

    def free_cell(self,pos:dpos, color:str):
        assert self._array[pos.x][pos.y]==color
        self._array[pos.x][pos.y]=None
        del self._color_list[color][pos]

    def fill_cell(self,pos:dpos, color:str):
        assert self._array[pos.x][pos.y] == None
        self._array[pos.x][pos.y] = color
        self._color_list.setdefault(color, dict())[pos] = 1

    def make_move(self,move):
        f=move.pos_from
        t=move.pos_to
        path = self._bg.check_path((f.x,f.y),(t.x,t.y))
        if not path:
            raise("no path")

        # assert self._array[f.x][f.y] == move.color.name
        # assert self._array[t.x][t.y]==None
        # self._array[f.x][f.y]=None
        # self._array[t.x][t.y]=move.color.name
        self.free_cell(move.pos_from,move.color.name)
        self.fill_cell(move.pos_to, move.color.name)
        self.current_move=None

        # self.update_graph()
        # self.update_buffer()

        return self.check_scrubs(move.pos_to)

    def make_history_move(self,move):
        f=move[0]
        t=move[1]
        self.free_cell(f,move[2])
        self.fill_cell(t, move[2])
        self.check_scrubs(t)
        self.scrub_cells()

    def update_graph(self):
        self._bg.update_graph(self)
        self.component_map = self._bg.get_components()

    def update_buffer(self):
        self._buffer=copy.deepcopy(self._array)

    # def prepare_view(self):
    #     rows = []
    #     for i in range(0,self._size):
    #         rows.append("  ".join([ '.' if x ==' ' else x for x in self._buffer[i]]))
    #     self.view = "\n".join(rows)

    def cell(self,pos:dpos):
        return self._array[pos.x][pos.y]

    def get_free_cells(self):
        free=[]
        for i in range(0,self._size):
            for j in range(0,self._size):
                if (self.cell(dpos(x=i,y=j))==None):
                    free.append(dpos(i,j))
        return free

    def get_random_free_cells(self,batch=None):
        free=self.get_free_cells()
        picked=[]
        batch = batch if batch else self._batch
        for i in range(0,batch):
            if len(free)<1:
                break
            pick = randint(0,len(free)-1)
            picked.append((free[pick], self._colors[randrange(self._colsize)]))
            free[pick]=free[-1]
            free.pop()
        return picked

    def valid(self,x,y):
        if (x>=0 and y>=0 and x<self._size and y<self._size):
            return True
        return False

    def valid_cell(self,cell:dpos)->bool:
        if (cell.x>=0 and cell.y>=0 and cell.x<self._size and cell.y<self._size):
            return True
        return False


    def get_scrubs_XY(self,pos:dpos):
        x,y=tuple(pos)
        scrubs = []
        color = self._array[x][y]
        if color == None:
            return scrubs
        for _dir in self._dirs[:4]:
            (dx, dy) = _dir
            dir=ddir(_dir[0],_dir[1])

            fw = self.get_segment_items(pos, dir)
            bw = self.get_segment_items(pos, ddir(-dir.dx,-dir.dy))

            fw_i=next((i for i, x in enumerate(fw) if x[1]!=color), len(fw))
            bw_i=next((i for i, x in enumerate(bw) if x[1]!=color), len(bw))

            if fw_i+bw_i-1 < self._scrub_length:
                continue

            scrub = [ v[0] for v in itertools.chain(reversed(bw[1:bw_i]), fw[0:fw_i])]
            scrubs.append(scrub)
            # (cx,cy)=(x,y)
            # while True:
            #     cx+=sx
            #     cy+=sy
            #     if not self.valid(cx,cy):
            #         break
            #     if self._array[cx][cy]==color:
            #         scrub.append(dpos(cx,cy))
            #     else:
            #         break
            # if len(scrub)>=self._scrub_length:
            #     # import pdb;pdb.set_trace()
            #     scrubs.append(scrub)
        return scrubs

    def check_scrubs(self, pos: dpos):
        # scrub_cells=[]
        self._scrubs = self.get_scrubs_XY(pos)

        # for i in range(0,self._size):
        #     for j in range(0, self._size):
        #         self._scrubs += self.get_scrubs_XY(i,j)
        return self._scrubs

    def check_tentative_scrubs(self, pos_from:dpos, pos_to:dpos, color):
        # scrub_cells=[]

        # self.free_cell(pos_from,color)
        self.fill_cell(pos_to, color)
        self._tentative_scrubs = self.get_scrubs_XY(pos_to)
        self.free_cell(pos_to, color)
        # self.fill_cell(pos_from, color)

        # for i in range(0,self._size):
        #     for j in range(0, self._size):
        #         self._scrubs += self.get_scrubs_XY(i,j)
        return self._tentative_scrubs

    def scrub_cells(self,scrubs=None):
        self._scrubs = scrubs if scrubs else self._scrubs
        scrubbed=dict()
        for scrub in self._scrubs:
            for item in scrub:
                if not item in scrubbed:
                    self.free_cell(item, self.get_cell(item))
                    scrubbed[item]=1
        self._scrubs=None
        # self.update_buffer()


    def get_history(self):
        return self._history

    def set_history(self, history, iteration=None):

        for item in history:
            for i,move  in enumerate(item['board']['move']):
                _move=(dpos.fromdict(move[0]), dpos.fromdict(move[1]), move[2]
                       , move[3], move[4], move[5])
                item['board']['move'][i]=_move
            _scrubs=[]
            for i,scrub  in enumerate(item['board']['remove']):
                _scrub=[]
                for cell in scrub:
                    assert len(cell)==2
                    if type(cell)==dict:
                        _scrub.append(dpos(cell['x'], cell['y']))
                    else:
                        _scrub.append(dpos(cell[0], cell[1]))
                _scrubs.append(_scrub)
            item['board']['remove']=_scrubs
            for i, nb in enumerate(item['board']['new']):
                _nb=(dpos.fromdict(nb[0]), nb[1])
                item['board']['new'][i]=_nb

        self._history=history
        if iteration==0:
            pass
        else:
            self.replay(iteration)

    def replay(self, iteration=None):
        self.reset()
        if iteration==None:
            iteration = len(self._history)
        else:
            iteration=int(iteration)
        last_state=None
        for i in range(iteration):
            history_move = self._history[i]
            for move in history_move['board']['move']:
                self.make_history_move(move)
            # for scrubs in history_move['board']['remove']:
            #     self.scrub_cells(scrubs)
            self.place(history_move['board']['new'])
            self.picked=history_move['board']['new']
            last_state=history_move['random_state']
        self.iteration=iteration
        if last_state:
            state=pickle.loads(base64.b64decode(last_state))
            random.setstate(state)
        self.history_post_load()
