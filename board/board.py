import copy
from heapq import *
import random
from random import randrange,randint
import collections
from collections import deque
import bisect
from sortedcontainers import SortedListWithKey, SortedDict
import pickle
import base64
import itertools
import math
import re

from typing import List, Set, Dict, Tuple, Text, Optional, AnyStr

from .utils import ddot,dpos,ddir,sign
from .board_graph import Board_graph
from .connect_assess import connect_assessor_c
from .position import position_c


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

##### constants

MOVE_FREE=0
MOVE_OCCUPIED=1
MOVE_BLOCKED=2
MOVE_UNBLOCKING=3
MOVE_UNBLOCKING_PASS=4

LOGLEVEL_NONE=0
LOGLEVEL_ERROR=1
LOGLEVEL_INFO=2
LOGLEVEL_DEBUG=3
LOGLEVEL_4=4


class move_c(ddot):
    cell_from:dpos
    cell_to:dpos

    def __init__(self, f:dpos, t:dpos):
        self.cell_from=f
        self.cell_to=t

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
    fake_prob=False

    _history=list()

    debug_hqueue=LOGLEVEL_INFO
    # debug_hqueue=LOGLEVEL_DEBUG
    debug_mio_changes=0
    check_mio=False
    debug_repeat = True
    debug_moves=[
        # (1,3,4,1),
        # (1,3,5,1),
        (7, 4 , 5, 8),
        (7, 4 , 5, 7),
        # (3,2,8,4),
        # ((1, 8), (0, 4))
    ]
    # axes: x - left, y - up, directions enumerated anti-clockwise

    def __init__(self, size=9, batch=3, colsize=None, scrub_length=5, axes=None, logfile=None, drawing_callbacks={}):
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
        self.free_cells = dict()
        self.free_cell_count=0
        # self.prepare_view()
        self.max_color_candidates=10
        self.max_free_moves=10
        self.max_obstacle_moves=10
        self.max_no_path_moves=100
        # self.components=list()
        self.component_map=dict()
        self.drawing_callbacks = drawing_callbacks
        self._logfile=logfile
        self.increases=[0]*(self._size+1)
        self.random_ahead=self._size*3
        self.random_queue=[]
        self.throw_known=True
        self.picked=None
        self.prepicked=None
        self.initial_batch=5
        self.cost_value_len = self._scrub_length * 2 + 4

        self.reset()
        self._bg.update_graph(self)

        if self._logfile:
            wh = open(self._logfile, 'w')
            wh.close()


    def set_debug_levels(self, debug):
        if 'hqueue' in debug:
            self.debug_hqueue=debug['hqueue']

    def reset(self):
        # COLORS=['Y','B','P','G','C','R','M']
        self._array = [
            [None for i in range(0, self._size)]
            for j in range(0, self._size)
        ]
        self.free_cell_count= self._size*self._size
        self.free_cells=dict()

        self._color_list = dict()
        self._scrubs = []
        self._bg = Board_graph(self._axes)
        self._pos_bg = Board_graph(self._axes)
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
                              , candidates=dict()
                              , cand_colors=SortedListWithKey(key=cand_color_sort)
                              , cand_color_moves=SortedListWithKey(key=lambda move: -move.total_gain)
                              , cand_color_map=dict()
                              , cand_map=dict()
                              , cand_cell_map=dict()
                              , cand_free_map=dict()
                              , no_path_map = dict()
                              , no_path_moves=SortedListWithKey(key=lambda move: -move.total_gain)
                              )
            # ddot(
            # move_map=dict()
            # , candidates=list()
            # , move_in=None, move_out=None, move_in_out=None
        # )

    def cell(self,cell:dpos) -> str:
        return self._array[cell.x][cell.y]

    # def set_cell(self,x,y,value):
    #     self._array[x][y]=value

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
            items.append([cell.copy(),self.cell(cell)])
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
            # else:
            #     cand.free[item_cell]={'lc':self._bg.metrics['lc'][tuple(item_cell)]}

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

    def candidates(self):
        A=self._assessment
        lines = self.get_chords()
        best_cand=None
        for line in lines:
            cell,dir,length,items=line
            for i in range(0, length-self._scrub_length+1):
                cand,colors = self.candidate(items[i][0],dir,items[i:i+self._scrub_length])
                # cand.index=len(self._cand_array)
                cand_key= self.cand_key(cand)
                A.candidates[cand_key]=cand
                for color in colors:
                    A.cand_colors.add(color)
                for cell in cand.cells:
                    A.cand_cell_map.setdefault(cell, dict())[self.cand_key(cand)] = cand

    def evaluation(self):
        A = self._assessment
        mio_counts = [ 0 for i in range(self._scrub_length * 2)]
        for cc in A.cand_colors:
            mio_counts[cc.move_in_out] += 1
        self.log("board mio: %s" % (mio_counts))
        return mio_counts

    def pos_evaluation(self, pos:position_c):
        A = self._assessment
        mio_counts = [ 0 for i in range(self._scrub_length * 2)]
        cmio_map=dict()
        for ckey,cand in A.candidates.items():
            cmio = self.comp_cand_mio(pos, cand)
            cmio_map[self.cand_key(cand)]=cmio
            for color,item in cmio.items():
                mio,count,ccells=item
                mio_counts[mio] += 1
        return mio_counts, cmio_map

    def get_mio(self, pos:position_c):
        pass

    def get_original_position(self)->position_c:
        A = self._assessment
        pos = position_c(self)
        for ckey, cand in A.candidates.items():
            self.cand_mio(pos,cand)
        self.update_position(pos)
        return pos

    def dependent(self, move1:ddot, move2:ddot)->bool:
        for cell1 in [move1.cell_from, move1.cell_to]:
            for cell2 in [move2.cell_from, move2.cell_to]:
                dx=cell1.x-cell2.x
                dy=cell1.y-cell2.y
                if dx==0 and abs(dy)<self._scrub_length:
                    return True
                if dy==0 and abs(dx)<self._scrub_length:
                    return True
                if abs(dx)==abs(dy) and abs(dx)<self._scrub_length:
                    return True
        return False

    def move_tuple(self,move):
        t= (move.cell_from.x, move.cell_from.y, move.cell_to.x, move.cell_to.y)
        return t


    def negate_tuple(self, t):
        if t==None:
            return None
        return tuple([-v for v in t])

    def picked_str(self,picked,prefix,suffix):
        str=''
        if picked:
            str = prefix+','.join(["%s%d.%d" % (item[1][0], item[0].x, item[0].y) for item in picked])+suffix
        return str

    def search_ahead(self):
        A=self._assessment
        original_position = self.get_original_position()
        A.position = self.get_original_position()
        A.best_moves = []
        unique = 0
        hqueue=[(None,-unique,None, original_position,[])]
        # queue=deque([(None,None,[])])
        # move_changes=dict()
        DEPTH=4
        skipped=0

        SIGNIFICANT=2

        max_value=None

        seen_trails=dict()

        best_value=None

        zero_value=[0]*(len(original_position.mio_counts)+4) # 1(in the middle) for metric
        assert len(zero_value)==self.cost_value_len

        scrubs=False

        position_count=0

        while hqueue:

            _value, _, move,position,trail=heappop(hqueue)
            value=self.negate_tuple(_value)
            if move!=None:
                assert len(value) == self.cost_value_len

            new_position = position
            worse = False
            if move!=None:
                if move.real_mio==None: # rough value estimate, assess and put back on queue
                    # if max_value !=None and value[0] < max_value:
                    #     continue
                    # tt = tuple(sorted([ self.move_tuple(move) for move in trail ]))
                    # if tt in seen_trails:
                    #     skipped+=1
                    #     continue
                    # seen_trails[tt]=1

                    if self.throw_known:
                        if len(trail)==1:
                            move.thrown=False
                    # if all([ self.move_tuple(trail[i] )== self.debug_moves[i] for i in range(len(trail))]):
                    #     if len(trail)==2:
                    #         debug=1
                    new_position,_scrubs = self.make_search_move(position, move, trail=trail)

                    position_count+=1
                    if _scrubs:
                        move.scrubs=True
                        scrubs=True
                    if self.debug_hqueue>=LOGLEVEL_DEBUG:
                        pos_str = "<%d %s %s q:%s" % ((max_value or 0), value, ",".join(
                            ["%s%d:%d,%d>%d,%d%s" % (move.color[0], move.move_type
                                                 , move.cell_from.x, move.cell_from.y
                                                 , move.cell_to.x, move.cell_to.y
                                                     , self.picked_str(move.picked, prefix='[', suffix=']')
                                                     )
                             for move in trail]
                        ), len(hqueue))
                        self.log(pos_str)

                    if (move.move_type==MOVE_UNBLOCKING
                            and not move.scrubs and len(trail)+1 < DEPTH):
                        move.real_mio = self._scrub_length
                        unique += 1
                        if self.debug_hqueue >= LOGLEVEL_DEBUG:
                            pos_str = "#%d %s %s q:%s p:%s" % ((max_value or 0), value, ",".join(
                                ["%s%d:%d,%d>%d,%d%s" % (move.color[0], move.move_type
                                                     , move.cell_from.x, move.cell_from.y
                                                     , move.cell_to.x, move.cell_to.y
                                                       , self.picked_str(move.picked, prefix='[', suffix=']')
                                                       )
                                 for move in trail]
                            ), len(hqueue), position_count)
                            self.log(pos_str)
                        move.move_type=MOVE_UNBLOCKING_PASS
                        heappush(hqueue, ((self.negate_tuple(value), -unique, move, new_position, trail)))
                    else:
                        diff = self.position_diff(original_position, new_position)

                        value,mio = tuple(self.rel_value(diff, len(trail), trail))
                        move.real_mio=self._scrub_length
                        if len(trail)>=DEPTH-1 or move.scrubs:
                            max_value=value[0] if max_value==None or value[0]>max_value else max_value
                        unique+=1
                        if self.debug_hqueue >= LOGLEVEL_DEBUG:
                            pos_str = ">%d %s %s q:%s p:%s" % ((max_value or 0), value, ",".join(
                                ["%s%d:%d,%d>%d,%d%s" % (move.color[0], move.move_type
                                                     , move.cell_from.x, move.cell_from.y
                                                     , move.cell_to.x, move.cell_to.y
                                                       , self.picked_str(move.picked, prefix='[', suffix=']')
                                                       )
                                 for move in trail]
                            ), len(hqueue), position_count)
                            self.log(pos_str)

                        heappush(hqueue, ((self.negate_tuple(value), -unique, move, new_position, trail)))
                    continue

                # move_changes[self.move_key(move)]=1
                if move.move_type!=MOVE_UNBLOCKING_PASS:
                    worse=False
                    val1=[]
                    val2, mio = self.gain_value(self.position_diff(original_position, new_position), len(trail)
                                                , trail)
                    if not A.best_moves:
                        worse=True
                    else:
                        val1,mio=self.gain_value(self.position_diff(original_position, A.position), len(A.best_moves)
                                            , A.best_moves)
                        # val2,mio=self.gain_value(self.position_diff(original_position, new_position), len(trail)
                        #                     , trail )
                        worse = val1 < val2
                    if worse==True:
                        A.best_moves=trail
                        A.position=new_position
                        # pos_str = "* " + pos_str
                        # self.log(pos_str)
                        if self.have_drawing_callback('assessment'):
                            self._bg.draw_moves(self,trail)
                            self.drawing_callback('assessment')
                        pos_str = "%d: %s%s %s %s q:%s p:%s" % ((max_value or 0), [' ', '*'][worse]
                                                                , [' ', 's'][bool(move) and move.scrubs]
                                                                , val2, ",".join(
                            ["%s%d:%d,%d>%d,%d%s" % (move.color[0], move.move_type
                                                   , move.cell_from.x, move.cell_from.y
                                                   , move.cell_to.x, move.cell_to.y
                                                     , self.picked_str(move.picked, prefix='[', suffix=']')
                                                     )
                             for move in trail]
                        ), len(hqueue), position_count)

                        self.log(pos_str)

            if not worse:
                pos_str = "<< %d: %s%s %s %s q:%s p:%s" % ((max_value or 0), [' ', '*'][worse]
                                          , [' ', 's'][bool(move) and move.scrubs]
                                          ,  value, ",".join(
                    ["%s%d:%d,%d>%d,%d%s" % (move.color[0], move.move_type
                                         , move.cell_from.x, move.cell_from.y
                                         , move.cell_to.x, move.cell_to.y
                                             , self.picked_str(move.picked, prefix='[', suffix=']')
                                             )
                     for move in trail]
                ), len(hqueue), position_count)

                self.log(pos_str)

            if move and move.scrubs: # not branching after a goal reached
                continue

            if max_value!=None:
                if value[0] < max_value-1:
                    break  # stopping because it's unlikely to yield higher results

            if move:
                if move.ccount >= len(position.color_list[move.color]):
                    continue # not branching as there are not enough colors

            # if value!=None and (best_value == None or best_value < value[0]):
            #     best_value = value[0]
            #
            # if scrubs or len(trail)+1>=DEPTH:
            #     if best_value > value[0]:
            #         break

            if len(trail)+1 < DEPTH:
                if value==None:
                    value=zero_value
                new_moves = self.find_new_moves(new_position,trail)
                for new_move in new_moves:
                    _trail=trail+[new_move]

                    if not len(value) == self.cost_value_len:
                        assert False
                    _diff=list(self.get_diff_from_value(value))
                    _diff[new_move.new_mio+2] += 1
                    diff=tuple(_diff)
                    estimate,mio = self.rel_value(diff, len(trail), trail)
                    unique += 1
                    if self.debug_hqueue >= LOGLEVEL_4:
                        pos_str = ">> %s %s q:%s p:%s" % (estimate, ",".join(
                            ["%s%s:%d,%d>%d,%d%s" % (move.color[0],move.move_type
                                                 , move.cell_from.x, move.cell_from.y
                                                 , move.cell_to.x, move.cell_to.y
                                                     , self.picked_str(move.picked, prefix='[', suffix=']')
                                                     )
                             for move in _trail]
                        ), len(hqueue), position_count)
                        self.log(pos_str)

                    heappush(hqueue, ((self.negate_tuple(estimate),-unique, new_move, new_position, _trail)))

        best_moves= self.rearrange(A.best_moves, original_position)
        if not best_moves:
            best_moves=A.best_moves
        # assert best_moves==A.best_moves
        return best_moves, A.position

    def rearrange(self, moves, original_position:position_c):
        cps = []
        seen_positions=dict()

        self.log("- %s " % (','.join(["%s:%d,%d>%d,%d" % ((original_position.cell(move.cell_from) or 'None')[0]
                                     , move.cell_from.x, move.cell_from.y
                                     , move.cell_to.x, move.cell_to.y)
                 for move in moves])))

        if not moves:
            return []
        if self.throw_known and len(moves)<2:
            return []

        trails=[]
        if self.throw_known:
            perm = itertools.permutations(range(1,len(moves)))
        else:
            perm = itertools.permutations(range(0,len(moves)))
        for _p in perm:
            if self.throw_known:
                p = (0, *_p)
            else:
                p= _p
            trail=[]
            cps=[]
            position = original_position
            for i, index in enumerate(p):
                self.update_position(position)

                cell=moves[index].cell_to
                if position.cell(cell)!=None:
                    path_exists=False
                elif position.cell(moves[index].cell_from)!=moves[index].color:
                    path_exists=False
                else:
                    path_exists, cross = self.check_pos_path_across(position
                                            ,moves[index].cell_from, moves[index].cell_to)
                if not path_exists:

                    cps=None
                    break


                if i==0:
                    cp=0
                else:

                    cp = self.assess_pos_cut_probability(position, moves[index])
                    if cp==None:
                        if self.debug_repeat:
                            cp = self.assess_pos_cut_probability(position, moves[index])
                cps.append(cp)
                tt = tuple(sorted([self.move_tuple(move[0]) for move in trail + [(moves[index],0)]]))
                new_position = seen_positions.get(tt)
                trail.append((moves[index], cp))
                if not new_position:
                    new_position, _scrubs = self.make_search_move(position, moves[index], trail=trail)
                    seen_positions[tt]=new_position
                position=new_position

            if not cps:
                continue
            prob = self.prob_sequence(cps)
            trails.append((prob, len(trails), trail))
        trails.sort()
        for prob,_,trail in trails:
            self.log("- %.4f %s " % ( prob, ','.join(["%s:%d,%d>%d,%d[%.4f]" % (
                                        (original_position.cell(move[0].cell_from) or 'None')[0]
                                        , move[0].cell_from.x, move[0].cell_from.y
                                        , move[0].cell_to.x, move[0].cell_to.y, move[1])
                                        for move in trail])))
        if not trails:
            return None
        return [ move_cp[0] for move_cp in trails[0][2]]


    # p1 + (1-p1)*p2 + (1- p1 - ((1-p1)*p2 )*p3 = p1 +p2 -p1p2 -p1p3 +p3 -p2p3+p1p2p3
    # 3p - 3p**2+p**3 = 1-(1-p)**3
    # probability of event occurence in a sequence of events with given probabilities
    def prob_sequence(self, probs):
        prob=0
        rest=1
        for p in probs:
            prob+=rest*p
            rest*=(1-p)
        assert round(prob+rest,6)==1
        return prob

    def comp_cand_mio(self, pos:position_c, cand:ddot):
        ccolors = collections.defaultdict(int)
        cmio = collections.defaultdict(int)
        free = 0
        balls = 0
        colors=dict()
        for ccell in cand.cells:
            ccolor = pos.cell(ccell)
            colors.setdefault(ccolor,dict())[ccell]=1
            if ccolor:
                ccolors[ccolor] += 1
                balls += 1
            else:
                free += 1
        for ccolor, count in ccolors.items():
            cmio[ccolor] = (free + (balls - count) * 2, count, colors[ccolor])
        return cmio

    def cand_mio(self, pos: position_c, cand):
        A=self._assessment
        cand_key=(cand.start, cand.dir)

        if pos.mio and cand_key in pos.mio:
            cmio = pos.mio[cand_key]
            return cmio

        if not pos.mio:
            pos.mio = dict()

        cmio = self.comp_cand_mio(pos,cand)

        # pos.mio[cand_key] = (cmio,cand.cells)
        pos.mio[cand_key] = cmio
        # for cell in cand.cells:
            # pos.mio_map.setdefault(cell, dict())[cand_key]=cmio
            # for ccolor, mio in cmio:
            #     pos.mio_slist.add((cand_key, ccolor, mio))
        return cmio

    def cost_filling(self, pos: position_c, cell: dpos, color, cand):
        cmio = self.cand_mio(pos, cand)
        assert pos.cell(cell)==None
        new_cmio = collections.defaultdict(int)
        changes=list()
        other_colors=0
        color_seen = False
        for ccolor, item in cmio.items():
            mio,count,ccells=item
            if ccolor == color:
                new_cmio[ccolor] = (mio - 1, count + 1)
                color_seen=True
            else:
                other_colors+=count
                new_cmio[ccolor]=(mio + 1,count)
            changes.append((mio, new_cmio[ccolor][0]))
        if not color_seen:
            new_cmio[color] = (self._scrub_length - 1 + other_colors, 1)
            changes.append((None, new_cmio[color][0]))
        if self.debug_mio_changes: # debug
            ck = self.cand_key(cand)
            self.log("o - %d,%d:%d,%d: %s" % (ck[0].x, ck[0].y, ck[1].dx, ck[1].dy, cmio))
            self.log("o + %d,%d:%d,%d: %s" % (ck[0].x, ck[0].y, ck[1].dx, ck[1].dy, new_cmio))

        return new_cmio,changes

    def cost_freeing(self, pos: position_c, cell: dpos, cand):
        cmio = self.cand_mio(pos, cand)
        color = pos.cell(cell)
        assert color!=None
        new_cmio = collections.defaultdict(int)
        changes=list()
        for ccolor, item in cmio.items():
            mio, count, ccells = item
            assert count >=1
            if ccolor==color:
                if count == 1: # color is going to be completely removed
                    new_cmio[ccolor]=(None, count-1)
                else:
                    new_cmio[ccolor] = (mio+1, count-1)
            else:
                new_cmio[ccolor]=(mio-1, count)

            changes.append((mio, new_cmio[ccolor][0]))
        if self.debug_mio_changes: # debug
            ck = self.cand_key(cand)
            self.log("x - %d,%d:%d,%d: %s"%(ck[0].x,ck[0].y,ck[1].dx,ck[1].dy, cmio))
            self.log("x + %d,%d:%d,%d: %s"%(ck[0].x,ck[0].y,ck[1].dx,ck[1].dy, new_cmio))

        return new_cmio,changes



    def update_when_freeing(self, pos:position_c, cell, scrubs=False):
        A=self._assessment
        src_cands = A.cand_cell_map[cell]
        for ckey,cand in src_cands.items():
            cmio, changes = self.cost_freeing(pos, cell, cand)
            for change in changes:
                if not scrubs and change[0]==0:
                    assert False
                # assert change[0]!=0
                assert change[1]!=0
                pos.mio_counts[change[0]] -= 1
                if not pos.mio_counts[change[0]]>=0:
                    assert False
                if change[1]!=None:
                    pos.mio_counts[change[1]] += 1

    def update_when_filling(self, pos:position_c, cell, color):
        A=self._assessment
        tgt_cands = A.cand_cell_map[cell]
        scrubs = []

        for ckey,cand in tgt_cands.items():
            cmio, changes = self.cost_filling(pos, cell, color, cand)
            for change in changes:
                assert change[0] != 0
                if change[0]!=None:
                    pos.mio_counts[change[0]] -= 1
                    if not pos.mio_counts[change[0]] >= 0:
                        assert False
                if change[1] == 0:
                    scrubs.append(cand)
                pos.mio_counts[change[1]] += 1
        return scrubs

    ## return mio_counts  - b.scrub*2 + 1
    def position_value(self, pos: position_c):
        # value = (pos.free_cell_count, pos.mio_counts)
        return (pos.free_cell_count, *pos.mio_counts)


    ## return free_cell_count+mio_counts diff adjusted for count of moves between two positions
    ## the greater the value, the earlier it is taken off the queue
    # diff == b.scrub*2 +1 # returns: b.scrub*2+3 : adjusted value + metrics placeholder
    def rel_value(self, diff, steps:int, trail:List[ddot]):
        if not len(diff)==self.cost_value_len-3:
            assert False

        disc_metric=None
        if trail:
            # comb_metric = round(sum([move.metric for move in trail]) * max([move.metric for move in trail]),4)
            disc_metric = trail[-1].metric
        else:
            disc_metric = -math.inf
            # comb_metric = -math.inf

        last_i = len(diff)
        for i in range(1,len(diff)):
            if diff[i] > 0:
                last_i=i
                break

        proj_mio = last_i-1
        proj_free_cells=diff[0]
        for i in range(0,3):
            if i< len(trail):
                if trail[i].scrubs==True:
                    break
            else:
                proj_mio-=1
                if proj_mio>0:
                    proj_free_cells-=self._batch
                else:
                    proj_free_cells+=self._scrub_length
                    break

        value = [ proj_free_cells, proj_mio, *diff[:self._scrub_length-1], -disc_metric, *diff[self._scrub_length-1:]]

        return value, 0

    def rel_value_(self, diff, steps:int, trail:List[ddot]):
        if not len(diff)==self.cost_value_len-1:
            assert False

        disc_metric=None
        if trail:
            # comb_metric = round(sum([move.metric for move in trail]) * max([move.metric for move in trail]),4)
            disc_metric = trail[-1].metric
        else:
            disc_metric = -math.inf
            # comb_metric = -math.inf

        value = [ *diff[:self._scrub_length-1], -disc_metric, *diff[self._scrub_length-1:]]

        return value, 0

    def get_diff_from_value(self,value):
        if not len(value)==self.cost_value_len:
            assert False
        return tuple([ *value[2:self._scrub_length+1], *value[self._scrub_length+2:]])


    def gain_value(self, diff, steps:int, trail:List[ddot]):
        return self.rel_value(diff,steps,trail)

    def gain_value_(self, diff, steps:int, trail:List[ddot]):
        if not len(diff)==self.cost_value_len-1:
            assert False
        v=0
        disc_metric=None
        if trail:
            # comb_metric = round(sum([move.metric for move in trail]) * max([move.metric for move in trail]),4)
            disc_metric = trail[-1].metric
        else:
            disc_metric = -math.inf
            # comb_metric = -math.inf


        value = [ *diff[:self._scrub_length-1], -disc_metric, *diff[self._scrub_length-1:]]

        return value, 0


    def rel_value_(self, _diff, steps:int, trail:List[ddot]):
        assert len(_diff)==self._scrub_length*2 +1
        v=0
        last_i=None
        diff = list(_diff)
        if trail:
            if trail[0].thrown==True:
                diff[0]+=self._batch
                assert diff[0]<=0
        for i in range(0, len(diff)):
            v+=sign(diff[i])
            if v>0:
                last_i=i
                break

        if trail:
            comb_metric = round(sum([move.metric for move in trail]) * max([move.metric for move in trail]),4)
        else:
            comb_metric = -math.inf


        if last_i==None:
            last_i=len(diff)
            value = [ -(len(diff)+steps), -comb_metric, *diff ]
        else:
            if last_i==0: # free_cell_count increase and mio_counts[0] increases actually mean the same: scrub
                last_i=1
            value = [-((last_i-1) + steps), -comb_metric, *diff]
        return value, last_i-1


    def gain_value_(self, _diff, steps:int, trail:List[ddot]):
        assert len(_diff)==self._scrub_length*2 +1
        v=0
        last_i=None
        diff=list(_diff)
        if trail:
            if trail[0].thrown == True:
                diff[0] += self._batch
                assert diff[0] <= 0

        for i in range(0, len(diff)):
            v+=sign(diff[i])
            if v>0:
                last_i=i
                break

        if trail:
            comb_metric = round(sum([move.metric for move in trail]) * max([move.metric for move in trail]),4)
        else:
            comb_metric = -math.inf
        if last_i==None:
            last_i=len(diff)
            value = [ -(len(diff)), -steps,  -comb_metric, *diff ]
        else:
            if last_i==0: # free_cell_count increase and mio_counts[0] increases actually mean the same: scrub
                last_i=1
            value = [-((last_i-1)), -steps, -comb_metric, *diff]
        return value, last_i-1

    # def position_rel_value(self, pos1: position_c, pos2: position_c, steps:int=0):
    #     diff = self.position_diff(pos1,pos2)
    #     return self.rel_value(diff,steps)

    ## return free_cell_count + mio_counts diff - b.scrub*2 +1
    def position_diff(self, pos1: position_c, pos2: position_c):
        val1= self.position_value(pos1)
        val2= self.position_value(pos2)
        assert len(val1) == self._scrub_length * 2 + 1
        assert len(val2) == self._scrub_length * 2 + 1
        t = tuple([y - x for x, y in zip(val1, val2)])
        # _val1 = (val1[0], *val1[1][1:]) # mio_counts[0] always 0
        # _val2 = (val2[0], *val2[1][1:])
        # t = tuple([y-x for x,y in zip(_val1, _val2)])
        return t

    ## compare free
    def pos_is_lesser(self, pos1:position_c, pos2:position_c, base:position_c=None):
        # if pos1==None or pos2==None:
        #     return None
        val1 = (pos1.free_cell_count, *self.position_value(pos1))
        val2 = (pos2.free_cell_count, *self.position_value(pos2))

        return val1<val2




    def cmp_cmio_map(self, map1, map2):
        diffs=list()
        seen=dict()
        for ckey in map1:
            seen[ckey]=1
            if ckey not in map2:
                diffs.append((ckey, map1[ckey],None))
            elif map1[ckey]!=map2[ckey]:
                diffs.append((ckey, map1[ckey], map2[ckey]))
        for ckey in map2:
            if ckey not in seen:
                diffs.append((ckey, None, map2[ckey]))
        return diffs

    def place_promised(self, picked, consume:bool, position=None):
        pos=position or self
        _picked = []
        _occupied = []
        for item in picked:
            if pos.cell(item[0]) == None:
                _picked.append(item)
            else:
                _occupied.append(item[1])

        pos.place(_picked)
        if len(_occupied) > 0:
            picked_add = self.get_random_free_cells(start=len(picked)
                                        , length=len(_occupied), position=pos)
            for i, color in enumerate(_occupied):
                picked_add[i] = (picked_add[i][0], color, picked_add[i][2])
            pos.place(picked_add)
            _picked += picked_add
        if consume:
            self.random_queue=self.random_queue[len(_picked):]
        return _picked



    def make_search_move(self, position:position_c, move:move_c, trail=None):
        A=self._assessment

        new_position = position.copy()

        # self.log("pos copy mio: %s" % (new_position.mio_counts))
        if self.check_mio:
            mio_counts,cmio_map1 = self.pos_evaluation(new_position)
        # self.log("pos check mio: %s" % (mio_counts))


        prepicked=[]
        if move.thrown==False:
            prepicked=self.prepicked
            assert prepicked
            # prepicked = self.get_random_free_cells(position=new_position)
            # if not self.prepicked==prepicked:
            #     assert False

        color = new_position.cell(move.cell_from)

        self.update_when_freeing(new_position,move.cell_from)

        color = new_position.free_cell(move.cell_from)

        if self.check_mio:
            mio_counts,cmio_map2 = self.pos_evaluation(new_position)
            diffs = self.cmp_cmio_map(cmio_map1, cmio_map2)
            if mio_counts!=new_position.mio_counts:
                assert False

        scrubs = self.update_when_filling(new_position, move.cell_to, color )
        new_position.fill_cell(move.cell_to, color)
        if self.check_mio:
            mio_counts,cmio_map3 = self.pos_evaluation(new_position)
            diffs = self.cmp_cmio_map(cmio_map2,cmio_map3)
            if mio_counts != new_position.mio_counts:
                assert False

        if scrubs:
            self.make_search_scrubs(new_position, scrubs)
        if self.check_mio:
            mio_counts,cmio_map4 = self.pos_evaluation(new_position)
            if mio_counts != new_position.mio_counts:
                assert False

        if not scrubs:
            if move.thrown==False:
                _picked = self.place_promised(prepicked, position=new_position, consume=False)
                move.picked=_picked

                # _picked=[]
                # _occupied=[]
                # for item in picked:
                #     if new_position.cell(item[0])==None:
                #         _picked.append(item)
                #     else:
                #         _occupied.append(item[1])
                #
                # new_position.place(_picked)
                # if len(_occupied) > 0:
                #     picked_add = new_position.get_random_free_cells(start=len(picked)
                #                 , length=len(_occupied), random_storage=self, consume=False)
                #     for i,color in enumerate(_occupied):
                #         picked_add[i]=(picked_add[i][0], color)
                #     new_position.place(picked_add)
                new_position.picked=_picked
                new_position.mio_counts, _ = self.pos_evaluation(new_position)
                move.thrown=True
            else:
                new_position.free_cell_count-=self._batch
        # pos_after_throw = position_c(self)
        # self.gifts(pos_before_throw, pos_after_throw)

        self.update_position(new_position)

        # p_dsize, p_pdsize = position.check_total_disc(move.cell_to)
        # m_dsize, m_pdsize = new_position.check_disc(move.cell_from)

        p_dsize = new_position.check_overall_disc()
        m_dsize = position.check_overall_disc()

        # self.log("new pos mio: %s" % (new_position.mio_counts))
        # move.metric=p_dsize-m_dsize+p_pdsize-m_pdsize
        move.metric = p_dsize-m_dsize
        return new_position, bool(scrubs)

    def update_position(self, position:position_c):
        position.update_max_colors()
        self._pos_bg.update_graph(position)
        position.metrics=copy.deepcopy(self._pos_bg.metrics)
        position.component_map,position.components = self._pos_bg.get_components()
        position.bi_component_map = self._pos_bg.get_bi_components()

    def make_search_scrubs(self, pos:position_c, scrubs: List[ddot]):
        A=self._assessment
        cells = dict()
        counter=0
        for cand in scrubs:
            for cell in cand.cells:
                if cells.setdefault(cell,counter)==counter:
                    self.update_when_freeing(pos, cell, scrubs=True)
                    color = pos.free_cell(cell)
                counter+=1

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

        best_moves, best_position = self.search_ahead()

        # self.color_assessment()
        # best_move = self.pick_best_move()
        # self.check_tentative_scrubs(best_move.pos_from,best_move.pos_to,best_move.color.name)
        if not best_moves:
            return None
        return best_moves[0]

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

    def adjusted_mio(self,pos,ckey):
        A=self._assessment
        _value = pos.mio[ckey]
        value={ k:v for k,v in _value.items()}
        cand = A.candidates[ckey]
        cells =cand.cells
        if self.prepicked:
            for item in self.prepicked:
                pcell,pcolor,rnd=item
                if pcell in cand.cells:
                    for color, item in value.items():
                        mio, count, ccells = item
                        if color==pcolor:
                            value[color]=(mio-1,count+1,ccells)
                        elif color!=None:
                            value[color]=(mio+1,count,ccells)
        return value

    def find_new_moves(self, pos: position_c, trail:List[ddot]) -> List[ddot]:

        A=self._assessment
        added = False

        new_moves:List[ddot]=[]
        blocked_moves:List[ddot]=[]
        obstacles:Dict[dpos,List[object]]=dict()

        def free_move_loop():
            nonlocal new_moves
            counter=0
            mio_list=[]
            A_cands = A.candidates
            for ckey,cand in A_cands.items():
                self.cand_mio(pos, cand)  # recomp if needed

            tgt_keys=None
            if trail:
                tgt_keys=trail[-1].cand_keys
            else:
                tgt_keys = pos.mio.keys()

            for tgt_key in tgt_keys:
                value = self.adjusted_mio(pos, tgt_key)
                # value=pos.mio[tgt_key]

                for color, item in value.items():
                    mio,count,ccells=item
                    mio_list.append((tgt_key, color, mio, count))
            # else:
            #     for cand_key, value in pos.mio.items():
            #         for color, item in value.items():
            #             mio,count,ccells=item
            #             mio_list.append((cand_key, color, mio, count))
            mio_slist = sorted(mio_list, key=lambda v: v[2])

            prepicked_cells=dict()
            if self.prepicked:
                prepicked_cells = dict([(item[0],1) for item in self.prepicked])

            # first_queue=[] # improvement expected
            # second_queue=[] # shuffling around, can be beneficial too
            _moves=[]
            for item in mio_slist: # every candidate,color, highest mio first
                cand_key, color, mio, count=item
                # mio=pos.mio[cand_key]
                cand = A.candidates[cand_key]
                cells=cand.cells
                for cell in cells:  # for candidate,color every free cell
                    ccolor = pos.cell(cell)
                    if ccolor == color:
                        continue
                    # if ccolor==None or True: # free
                    else:
                        if cell in prepicked_cells: # roughly, if we move the ball over a prepicked cell we negate its effects on this candidate
                            mio,count,_= pos.mio[cand_key][color]
                        # move_map.setdefault(cell,dict())
                        clist= pos.color_list[color]
                        exp_mio=None
                        for ccell in clist:  # for color,free cell every cell of the same color
                            # if ccell in move_map[cell]:
                            #     continue
                            if ccell in cells: # not shuffling
                                continue

                            if trail and ccell==trail[-1].cell_to:
                                continue
                            # move_map[cell][ccell]=1
                            exp_mio = mio - 1
                            if ccell in cells:
                                exp_mio=mio

                            move = ddot(cell_from=ccell, cell_to=cell, color=color
                                        , new_mio=exp_mio, real_mio=None, ccount=count+1
                                        , move_type=MOVE_FREE, scrubs=False
                                        , cand_keys=[ cand_key ]
                                        , thrown=None
                                        , picked=None
                                        , metric=0, total_gain=0, gain_detail=dict())

                            path_exists, cross = self.move_check(pos, move)
                            if path_exists:
                                move.metric= 0 # pos.metrics['lc'][tuple(move.cell_to)]
                                _moves.append((exp_mio, len(_moves), move))
                            elif cross:
                                move.move_type=MOVE_BLOCKED
                                for obcell in cross:
                                    obstacles.setdefault(dpos(obcell[0],obcell[1])
                                        , SortedListWithKey(key=lambda move:move.new_mio)).add(move)
                    # elif ccolor!=color:
                    #     move = ddot(cell_from=self.NOT_A_CELL, cell_to=cell, color=color
                    #                 , new_mio=mio - 2, real_mio=False, move_type=MOVE_OCCUPIED, scrubs=False
                    #                 , total_gain=0, gain_detail=dict())
                    #     obstacles.setdefault(cell,list()).append(move)
            _moves.sort()
            move_map=dict()
            counter=0
            for i, item  in enumerate(_moves):
                move = item[2]
                if move_map.setdefault(move.cell_from,dict()).setdefault(move.cell_to,i)==i:
                    new_moves.append(move)
                    counter+=1
                    if counter>=self.max_free_moves:
                        break
            return
        free_move_loop()


        def obst_move_loop():
            counter=0
            _moves=[]
            move_map=dict()

            move_keys = dict()

            obcells = sorted(obstacles.keys(), key=lambda obcell: obstacles[obcell][0].new_mio)

            # for obcell,moves in obstacles.items(): # every obstacle
            for ob_i, obcell in enumerate(obcells):
                moves = obstacles[obcell]
                if trail and obcell == trail[-1].cell_to:
                    continue

                move_mio=min([ move.new_mio for move in moves ])+1
                _move_mio = moves[0].new_mio+1
                if not move_mio==_move_mio:
                    assert False

                tgt_set=set()
                tgt_keys=[]
                min_mio=None
                move_it_key = moves.irange_key(max_key=max(1,moves[0].new_mio))

                for move in move_it_key:
                    for ckey in move.cand_keys:
                        if not ckey in tgt_set:
                            tgt_set.add(ckey)
                            tgt_keys.append(ckey)

                # move_map.setdefault(obcell,dict())
                obcolor = pos.cell(obcell)
                assert obcolor!=None
                clist = pos.color_list[obcolor]
                dispatched=False
                for ccell in clist:  # for obstacle : every cell of the same color
                    cands = A.cand_cell_map[ccell]
                    # for ckey, cand in cands.items():
                    #     self.cand_mio(pos,cand) # recomp if needed
                    # cands = pos.mio_map[ccell]
                    # cands = A
                    # assert len(A_cands)==len(cands)
                    for ckey,cand in cands.items(): # for cell of the same color : every candidate
                        cmio=self.cand_mio(pos,cand)
                        cmio = self.adjusted_mio(pos,ckey)

                        # cmio = pos.mio[self.cand_key(cand)]
                        _mio,count,cc_cells=cmio[obcolor]
                        cells=cand.cells
                        for cell in cells:   # for candidate : every free cell
                            if cell==obcell:
                                continue

                            exp_mio = _mio-1
                            if obcell in cc_cells:
                                exp_mio = _mio
                            if pos.cell(cell)==None:
                                move_tuple = (obcell, cell, tuple(tgt_keys))
                                # contra :
                                if move_tuple in move_map:
                                    continue
                                move_map[move_tuple]=1

                                exp_2mio = (move_mio,exp_mio)
                                if move_mio  > exp_mio:
                                    exp_2mio = (exp_mio, move_mio)
                                exp_mio=min(exp_mio,move_mio)
                                move = ddot(cell_from=obcell, cell_to=cell, color=obcolor
                                            , new_mio=exp_mio, real_mio=None, ccount=0
                                            , move_type=MOVE_UNBLOCKING
                                            , cand_keys = tgt_keys
                                            , thrown=None
                                            , picked=None
                                            , scrubs=False, metric=0, total_gain=0, gain_detail=dict())
                                path_exists, cross = self.move_check(pos, move)
                                if path_exists:
                                    move.metric = 0 # pos.metrics['lc'][tuple(move.cell_to)]
                                    _moves.append((exp_2mio, len(_moves), move))
                                    dispatched = True
                if dispatched:
                    continue
                for fcell in pos.free_cells:
                    mio=4
                    exp_2mio = (move_mio, mio)
                    move = ddot(cell_from=obcell, cell_to=fcell, color=obcolor
                                , new_mio=move_mio, real_mio=None, ccount=0
                                , move_type=MOVE_UNBLOCKING
                                , cand_keys=tgt_keys
                                , thrown=None
                                , picked=None
                                , scrubs=False, metric=0, total_gain=0, gain_detail=dict())
                    path_exists, cross = self.move_check(pos, move)
                    if path_exists:
                        move.metric = 0 # pos.metrics['lc'][tuple(move.cell_to)]
                        _moves.append((exp_2mio, len(_moves), move))
                        dispatched = True

            move_map = dict()

            _moves.sort()
            move_map = dict()
            counter = 0
            for i, item in enumerate(_moves):
                move = item[2]
                if move_map.setdefault(move.cell_from, dict()).setdefault(move.cell_to, i) == i:
                    new_moves.append(move)
                    counter += 1
                    if counter >= self.max_obstacle_moves:
                        break
            return
        obst_move_loop()

        return new_moves

    def assess_color_placement(self, cand_color) -> bool:

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
                                    if counter>self.max_obstacle_moves:
                                        return
            return
        obst_loop()
        return added

    def cand_key(self, cand:ddot):
        return (cand.start, cand.dir)

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
            src_dict[src_key]=src
            for tgt in tgt_cand_colors:
                if tgt.name != src.name:
                    continue
                tgt_key = self.color_cand_key(tgt)
                tgt_dict[tgt_key] = tgt
                sh = set([v.cell for v in src.cells ]) & set([v.cell for v in tgt.cells ])
                if sh:
                    shared.setdefault(src_key, dict())[tgt_key] =sh



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
                    for tgt_key,tgt_shared in shared_tgt.items():
                        tgt = tgt_dict.get(tgt_key)
                        if not tgt or move.pos_to not in tgt.cand.cells:
                            continue
                        # tgt_key=self.color_cand_key(tgt)
                        sh_mx=max(sh_mx, len(tgt_shared))
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
        move_key = (tuple(move.cell_from), tuple(move.cell_to))
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

    def board_cmp(self, mio1, mio2 ):
        return mio2 > mio1


    # def get_pos_components(self,pos):
    #
    #     pos.components  = self._pos_bg.get_components()


    def move_check(self, pos:position_c, move:ddot):
        # A = self._assessment
        # cand_color=move.color
        # move_key= self.move_key(move)
        # if move_key in A.move_map:
        #     return
        # if move_key in A.no_path_map:
        #     return
        #
        # gd = ddot(gain=0, loss=0, cand_cost=0, added_gain=0, src_gain=0, src_loss=0
        #                        , tgt_gain=0, tgt_loss=0, ob_block_cost=0
        #                        , lc=0, cut_prob=0)

        path_exists, cross = self.check_pos_path_across(pos, move.cell_from, move.cell_to)
        return path_exists, cross

    def add_move_candidate_(self, move:ddot, added_gain:float=0, blocked=None):
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

    def pos_cell_components(self, pos:position_c, cell:dpos):
        return pos.component_map.get(tuple(cell))

    def check_pos_path_across(self, pos:position_c, cell_from:dpos, cell_to:dpos):
        comps1 = self.pos_cell_components(pos, cell_from)
        if not comps1:
            return False, None

        comps2 = self.pos_cell_components(pos, cell_to)
        if not comps2:
            return False, None

        cross =set()
        path_exists=False
        if pos.cell(cell_to) !=None:
            for comp1 in comps1:
                for comp2 in comps2:
                    if comp1[0] == comp2[0]:
                        path_exists = False
                        cross|=set((tuple(cell_to),))
                        break
                else:
                    continue
                break
        elif pos.cell(cell_from) != None:
            assert len(comps2) == 1
            comp2 = comps2[0]
            for comp1 in comps1:
                if comp1[0] == comp2[0]:
                    path_exists = True
                else:
                    cross |= comp1[1] & comp2[1]
        else:
            comp2 = comps2[0]
            assert len(comps1)==1
            for comp1 in comps1:
                if comp1[0] == comp2[0]:
                    path_exists = True
                else:
                    cross |= comp1[1] & comp2[1]

        return path_exists, cross


    def check_path_across(self, cell_from, cell_to):
        comps1 = self.cell_components(cell_from)
        if not comps1:
            return False, None

        comps2 = self.cell_components(cell_to)
        assert len(comps2)==1
        comp2 = comps2[0]

        cross =set()
        path_exists=False
        if self.cell(cell_from) != None:
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
            ca = self._bg.assess_connection_wo_node(start_node, end_node, max_cut=3)
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


    def assess_pos_cut_probability(self, pos:position_c, move:ddot):
        start_node=(move.cell_from.x, move.cell_from.y)
        end_node=(move.cell_to.x, move.cell_to.y)
        # adj_free = self.free_adj_cells(move.cell_from)
        # start_edges= [ (start_node, tuple(adj) ) for adj in adj_free]
        if not self.fake_prob:
            self._pos_bg.change_free_graph(self, free_cells=[move.cell_from], fill_cells=[])
            ca = self._pos_bg.assess_connection_wo_node(start_node, end_node, max_cut=3)
            if ca==None:
                cut_prob=None
            else:
                cut_prob = ca.cut_probability()
            self._pos_bg.change_free_graph(self, free_cells=[], fill_cells=[move.cell_from])
            return cut_prob
        else:
            self._pos_bg.change_free_graph(self, free_cells=[move.cell_from], fill_cells=[])
            lc:float=None
            if self._pos_bg.check_path(start_node, end_node):
                lc = self._pos_bg.fake_assess_connection_wo_node(start_node, end_node, max_cut=3)
            self._pos_bg.change_free_graph(self, free_cells=[], fill_cells=[move.cell_from])

            if lc==None:
                return None
            cut_prob=1-lc
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
            cell:dpos = self.adj_cell(start_cell, dir)
            if not cell:
                continue
            color = self.cell(cell)
            cells.append(cell)
        return cells

    # def adj_cells(self, start_cell):
    #     cells=[]
    #     for dir in self._straight_dirs:
    #         npos:dpos = self.new_pos(start_cell, dir)
    #         if not npos:
    #             continue
    #         color = self.cell(npos)
    #         if color:
    #             continue
    #         cells.append(npos)
    #     return cells

    def next_move(self):
        history_item = dict(board=dict(move=list(), remove=list(), new=list()), random_state=None)
        scrubbed=False
        if self.iteration==0:
            _picked = self.get_random_free_cells(length=self.initial_batch)
        else:
            _picked = self.get_random_free_cells()
        if self.current_move:
            history_item['board']['move'].append((self.current_move.cell_from
                                        , self.current_move.cell_to
                                        , self.current_move.color
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
            pos_before_throw=position_c(self)
            if not self.prepicked:
                self.prepicked=_picked
            self.picked = self.place_promised(self.prepicked, consume=True) # actual placement
            self.prepicked=self.get_random_free_cells()
            # self.place(self.picked)
            pos_after_throw = position_c(self)
            self.gifts(pos_before_throw, pos_after_throw)
        else:
            debug=1

        history_item['board']['prepicked']=self.prepicked
        history_item['board']['new']=self.picked
        history_item['board']['random_queue']=self.random_queue
        history_item['random_state'] = base64.b64encode(pickle.dumps(random.getstate())).decode('ascii')

        if len(self._history) > self.iteration:
            self._history=self._history[:self.iteration]
        assert len(self._history) == self.iteration
        self._history.append(history_item)
        self.iteration+=1
        self.log("iteration: %d, luck stats: %s" % (self.iteration
                        , ','.join([ "%d:%d" % (i, v) for i,v in enumerate(self.increases) if v!=0 ])))
        # self._bg.update_graph(self)
        self.update_graph()

        if self.have_drawing_callback('after_throw'):
            self.draw()
            self.drawing_callback('after_throw')

        self.current_move = self.find_best_move()

        if self.current_move:
            if self.have_drawing_callback('move_found'):
                self.draw_move()
                self.drawing_callback('move_found')
        else:
            self.drawing_callback('over')
        return self.picked

    def color_counts(self, pos):
        A = self._assessment
        color_sets=collections.defaultdict(int)
        color_stats=[0]*(self._size+1)
        for ckey, cand in A.candidates.items():
            cmio = self.comp_cand_mio(pos, cand)
            for color, item in cmio.items():
                cells = item[2]
                cells_key=tuple(sorted([ tuple(cell) for cell  in cells]))
                if cells_key not in color_sets:
                    color_sets[cells_key]=1
                    color_stats[len(cells_key)]+=1
        return color_stats

    def gifts(self,pos1:position_c,pos2:position_c):
        stats1=self.color_counts(pos1)
        stats2=self.color_counts(pos2)
        for i in range(len(stats1)):
            d = stats2[i]-stats1[i]
            if i > 1 and d > 0:
                self.increases[i]+=d

    def drawing_callback(self,stage):
        cb = self.drawing_callbacks.get(stage, False)
        if cb == False:
            assert False
        if cb != None:
            cb(stage)

    def have_drawing_callback(self,stage):
        cb = self.drawing_callbacks.get(stage, False)
        if cb == False:
            assert False
        if cb != None:
            return True

    def history_post_load(self):
        print("iteration: %d" % (self.iteration))
        self.update_graph()
        if self.have_drawing_callback('after_throw'):
            self.draw()
            self.drawing_callback('after_throw')

        # self._bg.update_graph(self)
        if not self.prepicked:
            self.prepicked=self.get_random_free_cells()
        self.current_move = self.find_best_move()
        if self.current_move:
            if self.have_drawing_callback('move_found'):
                self.draw_move()
                self.drawing_callback('move_found')
        else:
                self.drawing_callback('over')
        return self.picked

    def draw(self, show=False):
        self._bg.init_drawing(self._axes)
        self._bg.draw(prepicked=self.prepicked)
        # if show:
        #     self._bg.show()

    def draw_move(self):
        self._bg.init_drawing(self._axes)
        self._bg.draw(prepicked=self.prepicked)
        start_edges=None
        tentative_scrub_edges=None
        if not self.current_move:
            f=t=None
        else:
            move=self.current_move
            f=move.cell_from
            t=move.cell_to
            start_node = (move.cell_from.x, move.cell_from.y)
            end_node = (move.cell_to.x, move.cell_to.y)
            adj_free = self.free_adj_cells(move.cell_from)
            start_edges = [(start_node, tuple(adj)) for adj in adj_free]
            if self._tentative_scrubs:
                ts= self._tentative_scrubs
                tentative_scrub_edges = [((ts[i].x, ts[i].y),(ts[i+1].x, ts[i+1].y)) for i in range(len(ts)-1)]
        self._bg.draw_move(self, f,t, tentative_scrub_edges)
        # self._bg.show()

    def draw_throw(self):
        self._bg.init_drawing(self._axes)
        self._bg.draw()
        # self._bg.show()

    def get_cells_by_color(self,color):
        return self._color_list[color]

    def place(self,picked):
        for item in picked:
            (pos,color,rnd)=item
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
        self.free_cell_count+=1
        del self._color_list[color][pos]

    def fill_cell(self,pos:dpos, color:str):
        assert self._array[pos.x][pos.y] == None
        self._array[pos.x][pos.y] = color
        self.free_cell_count-=1
        self._color_list.setdefault(color, dict())[pos] = 1

    def make_move(self,move):
        f=move.cell_from
        t=move.cell_to
        path = self._bg.check_path((f.x,f.y),(t.x,t.y))
        if not path:
            raise("no path")

        # assert self._array[f.x][f.y] == move.color.name
        # assert self._array[t.x][t.y]==None
        # self._array[f.x][f.y]=None
        # self._array[t.x][t.y]=move.color.name
        self.free_cell(move.cell_from,move.color)
        self.fill_cell(move.cell_to, move.color)
        self.current_move=None

        # self.update_graph()
        # self.update_buffer()

        return self.check_scrubs(move.cell_to)

    def make_history_move(self,move):
        f=move[0]
        t=move[1]
        self.free_cell(f,move[2])
        self.fill_cell(t, move[2])
        self.check_scrubs(t)
        self.scrub_cells()

    def update_graph(self):
        self._bg.update_graph(self)
        self.component_map, self.components = self._bg.get_components()
        self.bi_component_map = self._bg.get_bi_components()

    # def update_buffer(self):
    #     self._buffer=copy.deepcopy(self._array)

    # def prepare_view(self):
    #     rows = []
    #     for i in range(0,self._size):
    #         rows.append("  ".join([ '.' if x ==' ' else x for x in self._buffer[i]]))
    #     self.view = "\n".join(rows)

    # def cell(self,pos:dpos):
    #     return self._array[pos.x][pos.y]

    def get_free_cells(self):
        free=[]
        for i in range(0,self._size):
            for j in range(0,self._size):
                if (self.cell(dpos(x=i,y=j))==None):
                    free.append(dpos(i,j))
        return free

    def get_random_free_cells(self, start=0, length=None, position=None):
        pos = position or self
        free=pos.get_free_cells()
        picked=[]
        if length==None:
            length =self._batch
        # batch = batch if batch else self._batch
        assert start+length <= self.random_ahead
        if start+length > len(self.random_queue):
            # if not consume:
            #     assert False
            for i in range(len(self.random_queue), self.random_ahead):
                self.random_queue.append((randint(0,100000-1),randint(0,100000-1)))
        for i in range(start,start+length):
            pick = self.random_queue[i][0] % len(free)
            color = self._colors[self.random_queue[i][1] % self._colsize]
            # pick = randint(0,len(free)-1)
            picked.append((free[pick], color, self.random_queue[i]))
            free[pick] = free[-1]
            free.pop()
        return picked


    def get_random_free_cells_(self, start=0, length=None):
        free=self.get_free_cells()
        picked=[]
        if length==None:
            length =self._batch
        # batch = batch if batch else self._batch
        assert start+length <= self.random_ahead
        if start+length > len(self.random_queue):
            for i in range(len(self.random_queue), self.random_ahead):
                self.random_queue.append((randint(0,100000-1),randint(0,100000-1)))
        for i in range(start,start+length):
            pick = self.random_queue[i][0] % len(free)
            color = self._colors[self.random_queue[i][1] % self._colsize]
            # pick = randint(0,len(free)-1)
            picked.append((free[pick], color))
            free[pick] = free[-1]
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
                    self.free_cell(item, self.cell(item))
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
            self.picked = history_move['board']['new']
            for i, pick in enumerate(self.picked):
                _pick = list(pick)
                _pick[0]=dpos(_pick[0]['x'], _pick[0]['y'])
                _pick = tuple(_pick)
                self.picked[i] = _pick
                if len(_pick) != 3:
                    self.picked[i] = (_pick[0], _pick[1], 0)
            self.place(self.picked)
            self.prepicked = history_move['board']['prepicked']
            for i, pick in enumerate(self.prepicked):
                _pick = list(pick)
                _pick[0] = dpos(_pick[0]['x'], _pick[0]['y'])
                _pick = tuple(_pick)
                self.prepicked[i]=_pick
                if len(_pick) != 3:
                    self.prepicked[i] = (_pick[0], _pick[1], 0)

            # self.prepicked = history_move['board']['prepicked']
            last_state=history_move['random_state']
            if 'random_queue' not in history_move['board']:
                self.random_queue=[]
            else:
                self.random_queue = history_move['board']['random_queue']
        self.iteration=iteration
        if last_state:
            state=pickle.loads(base64.b64decode(last_state))
            random.setstate(state)
        self.history_post_load()


