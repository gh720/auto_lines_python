import copy
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

class position_c:
    id: int = 0
    array:List[List[str]]
    free_cells:Dict[dpos,object]
    free_cell_count:int
    mio_counts:List[int]
    mio : Dict[Tuple[dpos,ddir],object]
    color_list:Dict[dpos,str]
    mio_map: Dict[dpos,Dict[Tuple[dpos,ddir],object]]


    def __init__(self, board=None):
        self.id=0
        self.free_cells = dict()
        self.free_cell_count = 0
        self.mio = dict()
        self.mio_map = dict()
        if board:
            self.array=copy.deepcopy(board._array)
            self.find_free_cells()
            self.free_cell_count = len(self.free_cells)
            self.mio_counts = board.evaluation()
            mio_counts,cmio_map = board.pos_evaluation(self)
            if self.mio_counts!=mio_counts:
                assert False
            self.color_list = copy.deepcopy(board._color_list)

        else:
            self.color_list = None
            self.mio_counts = None

        # self.mio_slist=SortedListWithKey(key=lambda v: -v[2]) # sorted by -move_in_out

    def get_array(self):
        return self.array

    def get_size(self):
        return len(self.array)

    def find_free_cells(self):
        for i, ar in enumerate(self.array):
            for j, color in enumerate(ar):
                if color==None:
                    self.free_cells[dpos(i,j)]=1
        return

    def cell(self, cell:dpos)->str:
        return self.array[cell.x][cell.y]


    def clear_mio(self, cell:dpos):
        ckeys = self.mio_map[cell]
        for ckey in ckeys:
            del self.mio[ckey]
        del self.mio_map[cell]

    def free_cell(self,cell:dpos, color:str=None):
        _color = self.array[cell.x][cell.y]
        if color:
            assert _color ==color
        self.array[cell.x][cell.y]=None
        self.free_cell_count+=1
        assert cell !=None
        assert cell not in self.free_cells
        self.free_cells[cell]=1
        # self.mio_counts # taken care of in update_when_xxx
        del self.color_list[_color][cell]
        self.clear_mio(cell)
        return _color


    def fill_cell(self,cell:dpos, color:str):
        assert cell!=None
        assert self.array[cell.x][cell.y] == None
        self.array[cell.x][cell.y] = color
        self.free_cell_count-=1
        assert cell in self.free_cells
        del self.free_cells[cell]
        self.color_list.setdefault(color, dict())[cell] = 1
        self.clear_mio(cell)

    def copy(self):
        # assert isinstance(pos,cls)
        pos=self
        new_pos = position_c()
        position_c.id +=1
        new_pos.id=position_c.id
        new_pos.array = copy.deepcopy(pos.array)
        new_pos.free_cells=copy.deepcopy(pos.free_cells)
        new_pos.free_cell_count=pos.free_cell_count
        new_pos.color_list=copy.deepcopy(pos.color_list)
        new_pos.mio = copy.deepcopy(pos.mio)
        new_pos.mio_counts = copy.deepcopy(pos.mio_counts)
        new_pos.mio_map = copy.deepcopy(pos.mio_map)
        # new_pos.mio_slist = copy.deepcopy(pos.mio_slist)

        return new_pos

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
    fake_prob=True

    _history=list()

    debug_moves=[
        ((2,4),(0,5)),
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
        self.free_cells = None
        self.reset()
        self._bg.update_graph(self)
        # self.prepare_view()
        self.max_color_candidates=10
        self.max_free_moves=10
        self.max_obst_moves=10
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
        self.free_cells = self._size*self._size

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
                mio,count=item
                mio_counts[mio] += 1
        return mio_counts, cmio_map

    def get_mio(self, pos:position_c):
        pass

    def get_original_position(self)->position_c:
        A = self._assessment
        pos = position_c(self)
        for ckey, cand in A.candidates.items():
            self.cand_mio(pos,cand)
        self.update_pos_components(pos)
        return pos

    def search_ahead(self):
        A=self._assessment
        # for cand_color in A.color_cands:
        #     self.assess_color_placement(cand_color)
        original_position = self.get_original_position()
        A.position = self.get_original_position()
        A.best_moves = []
        queue=deque([(None,None,[])])
        DEPTH=4
        while queue:
            move,position,trail=queue.popleft()
            if move!=None:
                new_position = self.make_search_move(position, move)
                diff = self.position_diff(original_position, new_position )
                pos_str = "%s %s" % (",".join(
                    ["%d,%d>%d,%d" % (move.cell_from.x, move.cell_from.y, move.cell_to.x, move.cell_to.y)
                     for move in trail]
                ), diff)

                lesser = self.cmp_position(A.position, new_position)
                if lesser==True:
                    A.best_moves=trail
                    A.position=new_position
                    pos_str = "* " + pos_str
                else:
                    pos_str = "  " + pos_str
                self.log(pos_str)
            else:
                new_position=A.position

            if len(trail)+1 < DEPTH:
                new_moves = self.find_new_moves(new_position)
                for new_move in new_moves:
                    queue.append((new_move, new_position, trail+[new_move]))
        return A.best_moves, A.position

    def comp_cand_mio(self, pos:position_c, cand:ddot):
        ccolors = collections.defaultdict(int)
        cmio = collections.defaultdict(int)
        free = 0
        balls = 0
        for ccell in cand.cells:
            ccolor = pos.cell(ccell)
            if ccolor:
                ccolors[ccolor] += 1
                balls += 1
            else:
                free += 1
        for ccolor, count in ccolors.items():
            cmio[ccolor] = (free + (balls - count) * 2, count)
        return cmio

    def cand_mio(self, pos: position_c, cand):
        A=self._assessment
        cand_key=(cand.start, cand.dir)

        if pos.mio and cand_key in pos.mio:
            cmio = pos.mio[cand_key][0]
            return cmio

        if not pos.mio:
            pos.mio = dict()

        cmio = self.comp_cand_mio(pos,cand)

        pos.mio[cand_key] = (cmio,cand.cells)
        for cell in cand.cells:
            pos.mio_map.setdefault(cell, dict())[cand_key]=cmio
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
            mio,count=item
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
        if 0: # debug
            ck = self.cand_key(cand)
            self.log("- %d,%d:%d,%d: %s" % (ck[0].x, ck[0].y, ck[1].dx, ck[1].dy, cmio))
            self.log("+ %d,%d:%d,%d: %s" % (ck[0].x, ck[0].y, ck[1].dx, ck[1].dy, new_cmio))

        return new_cmio,changes

    def cost_freeing(self, pos: position_c, cell: dpos, cand):
        cmio = self.cand_mio(pos, cand)
        color = pos.cell(cell)
        assert color!=None
        new_cmio = collections.defaultdict(int)
        changes=list()
        for ccolor, item in cmio.items():
            mio, count = item
            assert count >=1
            if ccolor==color:
                if count == 1: # color is going to be completely removed
                    new_cmio[ccolor]=(None, count-1)
                else:
                    new_cmio[ccolor] = (mio+1, count-1)
            else:
                new_cmio[ccolor]=(mio-1, count)

            changes.append((mio, new_cmio[ccolor][0]))
        if 0: # debug
            ck = self.cand_key(cand)
            self.log("- %d,%d:%d,%d: %s"%(ck[0].x,ck[0].y,ck[1].dx,ck[1].dy, cmio))
            self.log("+ %d,%d:%d,%d: %s"%(ck[0].x,ck[0].y,ck[1].dx,ck[1].dy, new_cmio))

        return new_cmio,changes



    def update_when_freeing(self, pos:position_c, cell):
        A=self._assessment
        src_cands = A.cand_cell_map[cell]
        for ckey,cand in src_cands.items():
            cmio, changes = self.cost_freeing(pos, cell, cand)
            for change in changes:
                assert change[0]!=0
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

    def position_value(self, pos: position_c):
        value = (pos.free_cell_count, pos.mio_counts)
        return value

    def position_diff(self, pos1: position_c, pos2: position_c):
        val1= self.position_value(pos1)
        val2= self.position_value(pos2)
        _val1 = (val1[0], *val1[1])
        _val2 = (val2[0], *val2[1])
        t = tuple([y-x for x,y in zip(_val1, _val2)])
        return t

    def cmp_position(self, pos1:position_c, pos2:position_c):
        # if pos1==None or pos2==None:
        #     return None
        return self.position_value(pos1) < self.position_value(pos2)

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


    def make_search_move(self, position:position_c, move:move_c):
        A=self._assessment

        new_position = position.copy()

        # self.log("pos copy mio: %s" % (new_position.mio_counts))
        mio_counts,cmio_map1 = self.pos_evaluation(new_position)
        # self.log("pos check mio: %s" % (mio_counts))

        color = new_position.cell(move.cell_from)

        self.update_when_freeing(new_position,move.cell_from)

        color = new_position.free_cell(move.cell_from)
        mio_counts,cmio_map2 = self.pos_evaluation(new_position)
        diffs = self.cmp_cmio_map(cmio_map1, cmio_map2)
        if mio_counts!=new_position.mio_counts:
            assert False
        # src_cands= A.cand_cell_map[move.cell_from]
        #
        # for cand in src_cands:
        #     cmio,changes = self.cost_freeing(new_position, move.cell_from, cand)
        #     for change in changes:
        #         new_position.mio_counts[change[0]] -= 1
        #         new_position.mio_counts[change[1]] += 1


        scrubs= self.update_when_filling(new_position, move.cell_to, color )
        new_position.fill_cell(move.cell_to, color)
        mio_counts,cmio_map3 = self.pos_evaluation(new_position)
        diffs = self.cmp_cmio_map(cmio_map2,cmio_map3)
        if mio_counts != new_position.mio_counts:
            assert False

        # tgt_cands= A.cand_cell_map[move.cell_to]
        #
        # color = new_position.free_cell(move.cell_from)
        # new_position.fill_cell(move.cell_to, color)
        #
        # for cand in tgt_cands:
        #     cmio,changes = self.cost_filling(new_position, move.cell_to, color, cand)
        #     for change in changes:
        #         new_position.mio_counts[change[0]] -=1
        #         if change[1] <= 0:
        #             scrubs.append(cand)
        #         new_position.mio_counts[change[1]] += 1

        if scrubs:
            self.make_search_scrubs(new_position, scrubs)
        mio_counts,cmio_map4 = self.pos_evaluation(new_position)
        if mio_counts != new_position.mio_counts:
            assert False

        self.update_pos_components(new_position)
        # self.log("new pos mio: %s" % (new_position.mio_counts))
        return new_position

    def update_pos_components(self, position:position_c):
        self._pos_bg.update_graph(position)
        position.component_map = self._pos_bg.get_components()

    def make_search_scrubs(self, pos:position_c, scrubs: List[ddot]):
        A=self._assessment
        cells = dict()
        counter=0
        for cand in scrubs:
            for cell in cand.cells:
                if cells.setdefault(cell,counter)!=counter:
                    self.update_when_freeing(pos, cell)
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


    def find_new_moves(self, pos: position_c) -> List[ddot]:

        A=self._assessment
        added = False

        new_moves:List[ddot]=[]
        obstacles:Dict[dpos,List[object]]=dict()

        def free_move_loop():
            counter=0
            move_map=dict()
            mio_list=[]
            # A_cands = A.cand_cell_map[ccell]
            # for cand in A_cands:
            #     self.cand_mio(pos, ccell, cand)  # recomp if needed
            for cand_key, value in pos.mio.items():
                for color, mio in value[0].items():
                    mio_list.append((cand_key, color, mio))
            mio_slist = sorted(mio_list, key=lambda v: v[2])

            for item in mio_slist:
                cand_key, color, mio=item
                mio,cells = pos.mio[cand_key]
                for cell in cells:
                    ccolor = pos.cell(cell)
                    if ccolor==None: # free
                        move_map.setdefault(cell,dict())
                        clist= pos.color_list[color]
                        for ccell in clist:
                            if ccell in move_map[cell]:
                                continue
                            move_map[cell][ccell]=1
                            move = ddot(cell_from=ccell, cell_to=cell, color=color, total_gain=0, gain_detail=dict())
                            path_exists, cross = self.move_check(pos, move)
                            if path_exists:
                                new_moves.append(move)
                                counter+=1
                                if counter>=self.max_free_moves:
                                    return
                            else:
                                for obcell in cross:
                                    obstacles.setdefault(dpos(obcell[0],obcell[1]), list()).append(move)
                    elif ccolor!=color:
                        obstacles.setdefault(cell,list()).append(None)
            return
        free_move_loop()


        def obst_move_loop():
            counter=0
            move_map=dict()
            for obcell in obstacles:
                move_map.setdefault(obcell,dict())
                obcolor = pos.cell(obcell)
                assert obcolor!=None
                clist = pos.color_list[obcolor]
                dispatched=False
                for ccell in clist:
                    A_cands = A.cand_cell_map[ccell]
                    for ckey, cand in A_cands.items():
                        self.cand_mio(pos,cand) # recomp if needed
                    cands = pos.mio_map[ccell]
                    assert len(A_cands)==len(cands)
                    for ckey, cmio in cands.items():
                        cells = pos.mio[ckey][1]
                        for cell in cells:
                            if cell in move_map[obcell]:
                                continue
                            if pos.cell(cell)==None:
                                move_map[obcell][cell]=1
                                move = ddot(cell_from=obcell, cell_to=cell, color=obcolor, total_gain=0, gain_detail=dict())
                                path_exists, cross = self.move_check(pos, move)
                                if path_exists:
                                    new_moves.append(move)
                                    dispatched = True
                                    counter += 1
                                    if counter >= self.max_obst_moves:
                                        return
                if dispatched:
                    continue
                for fcell in pos.free_cells:
                    if fcell in move_map[obcell]:
                        continue
                    move_map[obcell][fcell]=1
                    move = ddot(cell_from=obcell, cell_to=fcell, color=obcolor, total_gain=0, gain_detail=dict())
                    path_exists, cross = self.move_check(pos, move)
                    if path_exists:
                        new_moves.append(move)
                        dispatched = True
                        counter+=1
                        if counter >= self.max_obst_moves:
                            return
        obst_move_loop()

        if not new_moves:
            assert False
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
                                    if counter>self.max_obst_moves:
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


    def get_pos_components(self,pos):

        pos.components  = self._pos_bg.get_components()


    def move_check(self, pos:position_c, move:ddot):
        A = self._assessment
        cand_color=move.color
        move_key= self.move_key(move)
        if move_key in A.move_map:
            return
        if move_key in A.no_path_map:
            return

        gd = ddot(gain=0, loss=0, cand_cost=0, added_gain=0, src_gain=0, src_loss=0
                               , tgt_gain=0, tgt_loss=0, ob_block_cost=0
                               , lc=0, cut_prob=0)

        path_exists, cross = self.check_pos_path_across(pos, move.cell_from, move.cell_to)

        return path_exists, cross
        #
        # if not path_exists:
        #     A.no_path_map[move_key] = move
        #     move.gain_detail = gd
        #     trimmed_cross = set()
        #     for node in cross:
        #         if self.get_cell(dpos(node[0], node[1])) != move.color.name:
        #             trimmed_cross.add(node)
        #
        #     move.cross = trimmed_cross
        #     A.no_path_moves.add(move)
        #     self.log("%s %d,%d>%d,%d: %.2f: g,l,cc,ag=%.2f,%.2f,%.2f,%.2f ob=%.2f cross=%s"
        #              % ('-', move.pos_from.x, move.pos_from.y, move.pos_to.x, move.pos_to.y
        #                 , move.total_gain, gd.gain, gd.loss, gd.cand_cost, gd.added_gain
        #                 , gd.ob_block_cost, cross))
        #     return True
        #
        # gd.gain =
        #
        # A.move_map[move_key] = move
        # A.cand_color_moves.add(move)
        #


#--------------

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
        assert len(comps2)==1
        comp2 = comps2[0]

        cross =set()
        path_exists=False
        if pos.cell(cell_from) != None:
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
            f=move.cell_from
            t=move.cell_to
            start_node = (move.cell_from.x, move.cell_from.y)
            end_node = (move.cell_to.x, move.cell_to.y)
            adj_free = self.free_adj_cells(move.cell_from)
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
        self.free_cells+=1
        del self._color_list[color][pos]

    def fill_cell(self,pos:dpos, color:str):
        assert self._array[pos.x][pos.y] == None
        self._array[pos.x][pos.y] = color
        self.free_cells-=1
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
        self.component_map = self._bg.get_components()

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
            self.place(history_move['board']['new'])
            self.picked=history_move['board']['new']
            last_state=history_move['random_state']
        self.iteration=iteration
        if last_state:
            state=pickle.loads(base64.b64decode(last_state))
            random.setstate(state)
        self.history_post_load()


