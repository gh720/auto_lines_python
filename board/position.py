from typing import List, Set, Dict, Tuple, Text, Optional, AnyStr
import copy, itertools, re
from collections import deque

from .utils import ddot,dpos,ddir,sign,prob_3


class cutset_c:
    cutsets:List[Tuple[Dict[dpos,int], Dict[dpos,object], Dict[dpos,object]]]
    def __init__(self,start,end,blocks, max=None):
        self.start=start
        self.end=end
        self.blocks=blocks
        self.cutsets = []
        self.MAX_CUTSET=max if max!=None else 3


class position_c:
    id: int = 0
    array:List[List[str]]
    free_cells:Dict[dpos,object]
    free_cell_count:int
    mio_counts:List[int]
    mio : Dict[Tuple[dpos,ddir],object]
    color_list:Dict[dpos,str]
    board:object
    # mio_map: Dict[dpos,Dict[Tuple[dpos,ddir],object]]


    _dirs = [
        [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]
    ]

    NOT_A_CELL=dpos(-1,-1)

    _straight_dirs=[]
    for dir in _dirs:
        if dir[0] and dir[1]:
            continue
        _straight_dirs.append(ddir(dir[0],dir[1]))


    def __init__(self, board=None):
        self.id=0
        self.free_cells = dict()
        self.free_cell_count = 0
        self.mio = dict()
        self.metrics=None
        self.max_colors=None
        self.component_map=None
        self.components=None
        # self.mio_map = dict()
        if board:
            self.board=board
            self.array=copy.deepcopy(board._array)
            self.find_free_cells()
            self.free_cell_count = len(self.free_cells)
            # self.mio_counts = board.evaluation()
            self.mio_counts,cmio_map = board.pos_evaluation(self)
            self._size=board._size
            # if self.mio_counts!=mio_counts:
            #     assert False
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
        ckeys = self.board._assessment.cand_cell_map[cell]
        for ckey in ckeys:
            del self.mio[ckey]

        # ckeys = self.mio_map[cell]
        # for ckey in ckeys:
        #     del self.mio[ckey]
        # del self.mio_map[cell]

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
        new_pos.board=self.board
        new_pos.mio=dict()
        new_pos.color_list=dict()

        new_pos.array = [
            [None for i in range(0, self.board._size)]
            for j in range(0, self.board._size)
        ]

        new_pos._size = self._size

        for i,ar in enumerate(pos.array):
            for j,color in enumerate(ar):
                new_pos.array[i][j]=color
                cell = dpos(i, j)
                if color ==None:
                    new_pos.free_cells[cell]=1
                else:
                    new_pos.color_list.setdefault(color, dict())[cell] = 1
        new_pos.free_cell_count=pos.free_cell_count

        for ckey in pos.mio:
            new_pos.mio[ckey]={ k:v for k,v in pos.mio[ckey].items() }
        new_pos.mio_counts = pos.mio_counts[:]
        # new_pos.mio = copy.deepcopy(pos.mio)
        # new_pos.mio_counts = copy.deepcopy(pos.mio_counts)
        # new_pos.mio_map = copy.deepcopy(pos.mio_map)
        # new_pos.mio_slist = copy.deepcopy(pos.mio_slist)

        return new_pos

    def update_max_colors(self):
        self.max_colors = max([ len(v) for k,v in self.color_list.items()],[0])

    def copy_(self):
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

    def adj_cell(self, cell:dpos, dir:ddir):
        d=dpos(cell.x+dir.dx, cell.y+dir.dy)
        if self.valid_cell(d):
            return d
        return None

    def valid_cell(self,cell:dpos)->bool:
        if (cell.x>=0 and cell.y>=0 and cell.x<self._size and cell.y<self._size):
            return True
        return False

    def check_overall_disc(self):
        comps = self.components
        total_disc_size=0
        if len(comps)<2:
            return 0
        for i,j in itertools.combinations(range(len(comps)),2):
            isec = comps[i][1] & comps[j][1]
            disc_size = (len(comps[i][1] - isec) * len(comps[j][0])
                         + len(comps[j][1] - isec) * len(comps[i][0]))
            total_disc_size += disc_size
        return total_disc_size

    def check_disc(self,cell, check_possible_disc=None):
        total_disc_size = 0
        total_possible_disc_size = 0
        bi_comps = self.bi_component_map.get(tuple(cell))
        if not bi_comps:
            return total_disc_size,total_possible_disc_size
        comps=[]

        if len(bi_comps) > 1:  # cut point
            neis = self.free_adj_cells(cell)
            t_neiset= set([tuple(cell) for cell in neis])
            for i, bi_comp in enumerate(bi_comps):
                t_node = list(t_neiset & bi_comp)[0]
                node=dpos(t_node[0],t_node[1])
                path,seen,bry = self.shortest_path_bry({node:0}, dict(), {cell:0}, dict())
                comps.append((set(seen), set(bry)))
            for i,j in itertools.combinations(range(len(comps)),2):
                isec = comps[i][1] & comps[j][1]
                disc_size = ( len(comps[i][1]-isec) * len(comps[j][0])
                              + len(comps[j][1]-isec) * len(comps[i][0]))
                total_disc_size += disc_size

            if not check_possible_disc:
                return total_disc_size, total_possible_disc_size


            for bci, bi_comp in enumerate(bi_comps):
                last_inner=last_outer=None
                for ni, nei in enumerate(neis):
                    if tuple(nei) in bi_comp and tuple(neis[(ni+1)%len(neis)]) not in bi_comp:
                        last_inner = nei
                    elif tuple(nei) not in bi_comp and tuple(neis[(ni+1)%len(neis)]) in bi_comp:
                        last_outer = neis[(ni+1)%len(neis)]
                if last_inner==last_outer:
                    continue
                # bi_neis = set([ nei for nei in neis if tuple(nei) in bi_comp])
                cs = cutset_c(last_inner, last_outer, blocks={cell: 0}, max=2)
                self.cut_rec2(cs, last_inner,last_outer, {cell:0}, dict())
                for cutset,ccomp,cbry in cs.cutsets:
                    possible_disc_size = 0
                    # t_start_neis=set([ tuple(nei) for nei in neis if nei not in ccomp])
                    start_neis=set([last_inner, last_outer])
                    t_neiset=set()
                    for node in cutset:
                        neis = self.free_adj_cells(cell)
                        t_neiset|=set([ tuple(nei) for nei in neis if not nei in cutset and nei in comps[bci][0] ])

                    comps2=[(set(ccomp),set(cbry))]
                    # nodes_seen:Set[Tuple[int,int]]=set()
                    nodes_seen=set(ccomp)
                    for t_nei in t_neiset:
                        nei = dpos(t_nei[0],t_nei[1])
                        if nei in nodes_seen:
                            continue
                        # a cut component taken for disc computation only if its node is in path or is not an articulation point
                        # otherwise the component is not new and the current cutset does not increase chances of cutting
                        if not (nei in start_neis or self.bi_component_map[t_nei]==1):
                            continue
                        path, seen, bry = self.shortest_path_bry({nei: 0}, dict(), cutset, dict())
                        comps2.append((set(seen),set(bry)))
                        nodes_seen|=set(seen)
                    for i, j in itertools.combinations(range(len(comps2)), 2):
                        isec = comps2[i][1] & comps2[j][1]
                        disc_size = (len(comps2[i][1] - isec) * len(comps2[j][0])
                                     + len(comps2[j][1] - isec) * len(comps2[i][0]))
                        possible_disc_size += disc_size
                    prob = prob_3(len(cutset), self.free_cell_count, 3)
                    total_possible_disc_size+=prob*possible_disc_size

        return total_disc_size, total_possible_disc_size

    def shortest_path(self,cells_from:Dict[dpos,int],cells_to:Dict[dpos,int],blocks:Dict[dpos,int]):
        queue=deque()
        seen = dict()
        assert not (set(cells_from)&set(cells_to))
        for cell in cells_from:
            queue.append([cell])
            # seen.setdefault(cell, len(seen))
        while queue:
            path = queue.popleft()
            cell = path[-1]
            assert isinstance(cell, dpos)
            if len(seen) != seen.setdefault(cell, len(seen)):
                continue
            neis = self.free_adj_cells(cell) # MAYBE: exclude=path[-1] to speed up
            for nei in neis:
                if nei in blocks:
                    continue
                if nei in cells_to:
                    return path+[nei], seen
                if nei in seen:
                    continue
                queue.append(path+[nei])
        return [], seen

    def shortest_path2(self,cells_from:Dict[dpos,int]
                       ,cells_to:Dict[dpos,int]
                       ,blocks:Dict[dpos,int]
                       ,seen:Dict[dpos,int]
                       ):
        queue=deque()
        # seen = dict()
        assert not (set(cells_from)&set(cells_to))
        for cell in cells_from:
            queue.append([cell])
            # seen.setdefault(cell, len(seen))
        while queue:
            path = queue.popleft()
            cell = path[-1]
            assert isinstance(cell, dpos)
            if len(seen) != seen.setdefault(cell, len(seen)):
                continue
            neis = self.free_adj_cells(cell) # MAYBE: exclude=path[-1] to speed up
            for nei in neis:
                if nei in blocks:
                    continue
                if nei in cells_to:
                    return path+[nei], seen
                if nei in seen:
                    continue
                queue.append(path+[nei])
        return [], seen

    def shortest_path_bry(self,cells_from:Dict[dpos,int],cells_to:Dict[dpos,int]
                          , blocks: Dict[dpos, int],seen:Dict[dpos,int]):
        queue=deque()
        bry = dict()
        assert not (set(cells_from)&set(cells_to))
        for cell in cells_from:
            queue.append([cell])
            # seen.setdefault(cell, len(seen))
        while queue:
            path = queue.popleft()
            cell = path[-1]
            assert isinstance(cell, dpos)
            if self.cell(cell) != None:
                bry.setdefault(cell, len(bry))
                continue
            elif len(seen) != seen.setdefault(cell, len(seen)):
                continue
            neis = self.adj_cells(cell)
            for nei in neis:
                if nei in blocks:
                    continue
                if nei in cells_to:
                    return path+[nei], seen, bry
                if nei in seen:
                    continue
                queue.append(path+[nei])
        return [], seen, bry

    def init_test_graph(self, text):
        if not text:
            text='''
            . y . . . r . . .
            r . . . . . . . p
            . y . p . . y y m
            c . r . . . . p .
            g b . p . . m . r
            . m . . . b . . .
            . . y . . . . c .
            . . . g . b . . .
            . . . . . g r . .
            '''
        self.reset()
        for i, str in enumerate(filter(lambda v: v!="", re.split("\r?\n", text))):
            for j, cc in enumerate(str.replace(' ','')):
                color = {k:v for k,v in map(lambda v: (v[0],v), self._colors)}.get(cc)
                if color!=None:
                    self.fill_cell(dpos(j,self._size-i-1),color)
                # self._array[j][self._size-i-1]=color
        return

    def dumps_test_graph(self):
        out='\n'
        for j in range(len(self._array)-1,-1,-1):
            line=' '.join([(self._array[i][j] or '.')[0] for i in range(len(self._array))])
            out=out+line+'\n'
        return out


    def cutoff(self, cell):
        neis=self.free_adj_cells(cell)
        # for i in range(len(neis)):
        #     for j in range(i+1,len(neis)):
        #         if not (abs(neis[i].x-neis[j].x)==2 or abs(neis[i].y-neis[j].y)==2):
        #             continue

        cs = cutset_c(neis[1],neis[3],blocks={cell:0})
        self.cut_rec(cs, cs.start, cs.end, cs.blocks)
        return cs

    def cutoff2(self, cell):
        neis=self.free_adj_cells(cell)
        # for i in range(len(neis)):
        #     for j in range(i+1,len(neis)):
        #         if not (abs(neis[i].x-neis[j].x)==2 or abs(neis[i].y-neis[j].y)==2):
        #             continue

        cs = cutset_c(neis[1],neis[3],blocks={cell:0})
        self.cut_rec2(cs, cs.start, cs.end, cs.blocks, dict())
        return cs


    def cut_rec(self,cs:cutset_c, start,end,blocks):
        path, seen = self.shortest_path({start:0}, {end:0}, blocks)
        # print("trying: %s"% ([ tuple(v) for v in blocks]))
        if not path:
            cs.cutsets.append(blocks)
        elif len(blocks)< cs.MAX_CUTSET:
            for i in range(1,len(path)-1):
                block=path[i]
                start=path[i-1]
                _blocks={ **blocks, **{block:len(blocks)} }
                self.cut_rec(cs, start, end, _blocks)

    def cut_rec2(self,cs:cutset_c, start,end,blocks, comp):
        path, path_comp, bry = self.shortest_path_bry({start:0}, {end:0}, blocks, { **comp } )
        _blocks=blocks
        # print("trying: %s"% ([ tuple(v) for v in blocks]))
        if not path:
            cs.cutsets.append((blocks, path_comp, bry))
            return path_comp
        elif len(blocks)< cs.MAX_CUTSET:
            current_comp = { **comp }
            for i in range(1,len(path)-1):
                block=path[i]
                start=path[i-1]
                _blocks={ **blocks, **{block:len(blocks)} }
                _comp = self.cut_rec2(cs, start, end, _blocks, current_comp)
                if _comp:
                    current_comp=_comp
            return dict()
        return dict()

    def cutoff_these(self, nei1, nei2, blocks=dict()):
        path=[]

        queue=deque()
        while queue:
            path,comp1,comp2 = queue.popleft()
            for i, node in enumerate(path):

                path, seen = self.shortest_path( node
                                                , { node: 1 for node in path[i+1:]}
                                                , { **blocks, node:len(blocks)})
                if not path:
                    comp1.update(seen)
                    if not comp2:
                        _, comp2 = self.shortest_path({ node: 1 for node in path[i+1:]}
                                                      , { node: 1 for node in path[:i]}
                                                      , blocks)
                    else:
                        all(map(lambda k: comp2.pop(k,None), blocks))
                else:
                    pass
                _blocks = {k: v for k, v in blocks.items()}
                _blocks[node] = len(_blocks)
                queue.append((_blocks,comp1,comp2))


    def cutoff_these_(self, nei1, nei2, max_cutoff_len=3, blocks=dict()):
        queue = deque([blocks])
        min_cutoff_len=None
        cutoffs=[]
        while queue:
            blocks,comp1,comp2 = queue.popleft()
            path,seen = self.shortest_path({nei1:0}, {nei2:0}, blocks)
            if not path:
                if min_cutoff_len==None or len(blocks) < min_cutoff_len:
                    min_cutoff_len=len(blocks)
                if len(blocks)<=max_cutoff_len:
                    cutoffs.append(blocks)
                cut_comp1 = { **cut_comp1, **seen }
                if not cut_comp2:
                    _, cut_comp2 = self.shortest_path({nei2: 0}, {nei1: 0}, blocks)
                else:
                    all(map(cut_comp2.pop, cut_comp1))

            else:
                for i, node in enumerate(path):

                    path, seen = self.shortest_path({ node: 1 for node in path[:i]}
                                                    , { node: 1 for node in path[i+1:]}
                                                    , { **blocks, node:len(blocks)})
                    if not path:
                        comp1=seen
                        if not cut_comp2:
                            _, cut_comp2 = self.shortest_path({nei2: 0}, {nei1: 0}, blocks)
                        else:
                            all(map(cut_comp2.pop, cut_comp1))
                    else:
                        pass
                    _blocks = {k: v for k, v in blocks.items()}
                    _blocks[node] = len(_blocks)
                    queue.append((_blocks,comp1,comp2))

        return cutoffs
