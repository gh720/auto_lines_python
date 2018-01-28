import copy 
from random import randrange,randint
import collections
import bisect
from sortedcontainers import SortedListWithKey

from .utils import ddot,dpos,ddir
from .board_graph import Board_graph
from .connect_assess import connect_assessor_c


'''
what can be done:
1. placing at adj free cells
2. -- (4, 5) to (4, 3): 0.000000 - should not be (max_edge_cut not big enough ?)
3. should have assessed (0,5 - 4,5) candidate first
'''


class Board:
    # COLORS=['Y','B','P','G','C','R','M']
    _array=[]
    _size=None
    _colors="red green yellow blue purple cyan magenta".split(' ')
    _colsize=len(_colors)
    _color_list=dict()
    _scrap_length=None
    _scraps=[]
    _bg=Board_graph()
    current_move = None
    _assessment=None # ddot(moves=list(), move_map=dict(), candidates=SortedListWithKey(key=lambda cand: cand.move_in_out))
    _axes= None


    # axes: x - left, y - up, directions enumerated anti-clockwise
    _dirs = [
        [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]
    ]

    _straight_dirs=[]
    for dir in _dirs:
        if dir[0] and dir[1]:
            continue
        _straight_dirs.append(ddot(dx=dir[0],dy=dir[1]))

    def __init__(self, size=9, batch=5, colsize=None, scrap_length=5, axes=None):
        if colsize==None:
            colsize=len(self._colors)
        self._size=size
        self._batch=batch
        self._scrap_length=scrap_length
        self._colsize=colsize
        self._sides=[[0,None], [None,0], [self._size-1,None], [None,self._size-1]]
        self._array = [
            [None for i in range(0,self._size)]
                for j in range(0,self._size)
        ]
        self._color_list=collections.defaultdict(dict)
        self._axes = axes
        self._bg=Board_graph(self._axes)
        self.update_buffer()
        self.reset_assessment()
        self._bg.update_graph(self)
        # self.prepare_view()

    def get_array(self):
        return self._array

    def get_size(self):
        return self._size

    def reset_assessment(self):
        self._assessment=ddot(moves=list()
                              , move_map=dict()
                              , candidates=list()
                              , cand_colors=SortedListWithKey(key=lambda color: color.move_in_out)
                              )
            # ddot(
            # move_map=dict()
            # , candidates=list()
            # , move_in=None, move_out=None, move_in_out=None
        # )

    def get_cell(self,x,y):
        return self._array[x][y]

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

    def sign(x):
        if x==0:
            return 0
        return -1 if x<0 else 1

    def get_segment(self,x,y,dx,dy):
        line=None
        length=None
        for side in self._sides:
            hit = self.ray_hit(x,y,dx,dy,side)
            if hit==None:
                continue
            x0,y0=hit
            if dx!=sign(x0-x):
                continue
            if dy!=sign(y0-y):
                continue
            _mx = max(abs(y0-y), abs(x0-x))
            if length==None or _mx < length:
                line=[x,y,dir_index, length ]
        assert line!=None
        return line

    def get_line(self, x,y, dx,dy):
        line = self.get_segment(x,y,dx,dy)
        line.append(self.get_segment_items(x,y,dx,dy,length))
        return line


    # returns x,y and length of a board's chord for given dx,dy
    def get_starts(self,dx,dy):
        starts=None
        if dx==0:
            starts=[(x,0 if dy>0 else self._size-1,self._size) for x in range(0,self._size)]
        elif dy==0:
            starts=[(0 if dx>0 else self._size-1, y, self._size) for y in range(0,self._size)]
        elif dx==dy:
            count = max(0,self._size-self._scrap_length+1)
            starts=([ (0, y, self._size-y) if dx>0 else (self._size-1, self._size-y-1, self._size-y) # vert for /
                        for y in range(0,count) ] + 
                    [ (x, 0, self._size-x) if dx>0 else (self._size-x-1, self._size-1, self._size-x) # horiz for /
                        for x in range(1,count) ] )
        elif dx==-dy:
            count = max(0,self._size-self._scrap_length+1)
            starts=([ (0, self._size-y-1, self._size-y) if dx>0 else (self._size-1, y, self._size-y) # vert for \
                        for y in range(0,count) ] + 
                    [ (x, self._size-1, self._size-x) if dx>0 else (self._size-x-1, 0, self._size-x) # horiz for \
                        for x in range(1,count) ])
        assert starts!=None
        return starts


    def get_segment_items(self,x,y,dx,dy,length):
        items=[]
        cx,cy=x,y
        for l in range(length):
            items.append([self.get_cell(cx,cy),cx,cy])
            cx+=dx
            cy+=dy
        return items

    def get_chords(self):
        lines=[]
        for k in range(0,4):
            dx,dy=self._dirs[k]
            starts=self.get_starts(dx,dy)
            for x,y,length in starts:
                items=self.get_segment_items(x,y,dx,dy,length)
                lines.append([x,y,dx,dy,length,items])
        return lines


    def candidate(self,x,y,dx,dy,items):
        cand = ddot(colors=dict(),free=dict(),poses=dict(), ball_count=0, move_in_out=0
                    , index=None, rank=None, moves=dict())
        # colors = dict()
        # colors = AttrDict(color=AttrDict(pos=list(),count=0))
        # free=dict()
        # ball_count=0
        for i in range(len(items)):
            color,cx,cy=items[i]
            cand.poses[dpos(cx,cy)]=color
            if color!=None:
                cand.colors.setdefault(color, ddot(pos=list(),name=color,count=0,cand=cand, moves=dict()))
                cand.colors[color].pos.append(ddot(i=i, pos=dpos(cx,cy)))
                cand.colors[color].count+=1
                cand.ball_count+=1
            else:
                cand.free[dpos(cx,cy)]={'lc':self._bg.metrics['lc'][(cx,cy)]}

        cand_colors = []
        for i_color, color in cand.colors.items():
            color.move_in_out = len(cand.free) + (cand.ball_count - color.count)
            cand_colors.append(color)

        return cand,cand_colors


    def candidates(self):
        A=self._assessment
        lines = self.get_chords()
        best_cand=None
        for line in lines:
            x,y,dx,dy,length,items=line
            for i in range(0, length-self._scrap_length+1):
                cand,colors = self.candidate(x,y,dx,dy,items[i:i+self._scrap_length])
                # cand.index=len(self._cand_array)
                A.candidates.append(cand)
                for color in colors:
                    A.cand_colors.add(color)


    def color_assessment(self):
        counter=0
        for color in self._assessment.cand_colors:
            self.assess(color)
            counter+=1
            if counter>20:
                break


    def find_best_move(self):
        self.reset_assessment()
        self.candidates()
        self.color_assessment()
        return self.pick_best_move()

    def pick_best_move(self):
        A=self._assessment
        best_move=None
        for cand in A.candidates:
            for move in cand.moves:
                best_move= move
                break
        else:
            return None

        return move,cand


    def assess(self,color):
        A=self._assessment
        self.assess_color_placement(color)

    def add_move_candidate(self, move, color):
        A=self._assessment
        move_key = (move.pos_from, move.pos_to)
        if move_key in A.move_map:
            return

        cut_prob = self.assess_move(move)
        move.cut_prob=cut_prob
        A.move_map[move_key] = move
        color.moves[move_key] = move

    def assess_color_placement(self, color):

        # A=self._assessment
        counter=0
        if color.cand.free:
            for free_pos, metric in color.cand.free.items():
                color_cells = self.get_cells_by_color(color.name)
                for pos in color_cells:
                    if pos in color.cand.poses:
                        continue
                    move = ddot(pos_from=pos, pos_to=free_pos, color=color)
                    self.add_move_candidate(move, color)
                    counter+=1
                    if counter>3:
                        break
                else:
                    continue
                break


    # might be not necessary
    def assess_cand_placement(self, cand):

        # A=self._assessment
        if cand.free:
            scolors=sorted(cand.colors.items(), key=lambda i:i[1].count, reverse=True)
            for i_color, color in scolors:
                for free_pos, metric in cand.free.items():
                    color_cells = self.get_cells_by_color(i_color)
                    for pos in color_cells:
                        if pos in cand.poses:
                            continue
                        move = ddot(pos_from=pos, pos_to=free_pos, color=color)
                        self.add_move_candidate(move, cand)

    def assess_move(self, move:ddot):
        # adj_cells= self.free_adj_cells(start_pos)
        # for cell in adj_cells:
        #     start_node = cell.x

        start_node=(move.pos_from.x, move.pos_from.y)
        end_node=(move.pos_to.x, move.pos_to.y)
        adj_free = self.free_adj_cells(move.pos_from)
        start_edges= [ (start_node, tuple(adj) ) for adj in adj_free]
        ca = self._bg.assess_connection_wo_node(start_node, end_node, start_edges, max_cut=3)
        ass = ca.cut_probability()
        print ("%s to %s: %f" % (start_node, end_node, ass))
        return ass

    def new_pos(self, pos:dpos, dir:ddir):
        d=dpos(pos.x+dir.dx, pos.y+dir.dy)
        if self.check_pos(d):
            return d
        return None

    def check_pos(self, pos:dpos):
        if pos.x<0 or pos.y<0 or pos.x>=self._size or pos.y>=self._size:
            return False
        return True

    def free_adj_cells(self, start_pos):
        cells=[]
        for dir in self._straight_dirs:
            npos:dpos = self.new_pos(start_pos, dir)
            if not npos:
                continue
            color = self.cell(npos)
            if color:
                continue
            cells.append(npos)
        return cells


    def candidates_(self):
        # hash of cand.lines by cell pos
        # cand.lines have obstacle info
        
        lines = self.get_chords()
        all_candidates = []
        for line in lines:
            x,y,dx,dy,length,items=line
            cand=collections.defaultdict(lambda : collections.defaultdict(dict))
            for i in range(length):
                color,cx,cy=items[i]
                if color==None:
                    continue
                # segment= self.get_segment(x,y,dx,dy)
                for j in range(max(i-self._scrap_length+1,0), min(i, length-self._scrap_length)+1):
                    cand[color][j][i-j]=(cx,cy)
            if len(cand):
                all_candidates.append([x,y,dx,dy,length,cand])
        return all_candidates

    def picked_balls(self,picked):
        msg=",".join(["%s[%s,%s]" % (picked[i][2],picked[i][0],picked[i][1]) for i in range(0,len(picked))])
        return msg

    def next_move(self):
        if self.current_move:
            self.make_move(self.current_move)
        self.picked= self.get_random_free_cells()
        self.place(self.picked)
        self.last_balls= self.picked_balls(self.picked)
        self._bg.update_graph(self)
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
        if not self.current_move:
            f=t=None
        else:
            move,cand=self.current_move
            f=move.pos_from
            t=move.to_pos
        self._bg.draw_move(self, f,t)
        self._bg.show()

    def get_cells_by_color(self,color):
        return self._color_list[color]

    def place(self,picked):
        for item in picked:
            (x,y,color)=item
            self._array[x][y]=color
            self._color_list.setdefault(color,dict())[dpos(x,y)]=1
            # self.color_lists[color]
        self.update_graph()
        self.update_buffer()

    def make_move(self,move,cand):
        f=move.pos_from
        t=move.pos_to
        path = self._bg.check_path((f.x,f.y),(t.x,t.y))
        if not path:
            raise("no path")

        assert self._array[f.x][f.y] == move.color
        assert self._array[t.x][t.y]==None
        self._array[f.x][f.y]=None
        self._array[t.x][t.y]=move.color
        self.check_scraps(move.pos_to)

        self.update_graph()
        self.update_buffer()


    def update_graph(self):
        self._bg.update_graph(self)

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
                if (self.cell(ddot(x=i,y=j))==None):
                    free.append([i,j,self._colors[randrange(self._colsize)]])
        return free

    def get_random_free_cells(self,batch=None):
        free=self.get_free_cells()
        picked=[]
        batch = batch if batch else self._batch
        for i in range(0,batch):
            if len(free)<1:
                break
            pick = randint(0,len(free)-1)
            picked.append(free[pick])
            free[pick]=free[-1]
            free.pop()
        return picked

    def valid(self,x,y):
        if (x>=0 and y>=0 and x<self._size and y<self._size):
            return True
        return False

    def get_scraps_XY(self,x,y):
        # _dirs = [
        #     [1,0], [1,1], [0,1], [-1,1]
        # ]
        scraps = []
        color = self._array[x][y]
        if color == None:
            return scraps
        for dir in self._dirs[:4]:
            scrap = [[x,y]]
            (sx,sy)=dir
            (cx,cy)=(x,y)
            while True:
                cx+=sx
                cy+=sy
                if not self.valid(cx,cy):
                    break
                if self._array[cx][cy]==color:
                    scrap.append([cx,cy])
                else:
                    break
            if len(scrap)>=self._scrap_length:
                # import pdb;pdb.set_trace()
                scraps.append(scrap)
        return scraps

    def check_scraps(self,pos):
        # scrap_cells=[]
        self.scraps=[]
        for i in range(0,self._size):
            for j in range(0, self._size):
                self.scraps += self.get_scraps_XY(i,j)
        return self.scraps

    def get_slope(self,scrap):
        f = scrap[0]
        l = scrap[-1]
        w = l[0]-f[0]
        h = l[1]-f[1]
        if w ==0:
            slope= '-'
        elif h==0:
            slope='|'
        elif w==h:
            slope='\\'
        elif w==-h:
            slope='/'
        return slope

    def draw_scrap(self,scrap):
        slope = self.get_slope(scrap)
        for item in scrap:
            self._buffer[item[0]][item[1]]=slope


    def draw_scraps(self,scraps=None):
        self.scraps = scraps if scraps else self.scraps
        for scrap in self.scraps:
            self.draw_scrap(scrap)
        # self.prepare_view()
        self.scrap_cells()

    def scrap_cells(self,scraps=None):
        self.scraps = scraps if scraps else self.scraps
        for scrap in self.scraps:
            for item in scrap:
                self._array[item[0]][item[1]]=None
        # self.update_buffer()

