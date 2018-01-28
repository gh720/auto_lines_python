import copy 
import random
from random import randrange,randint
import collections
import bisect
from sortedcontainers import SortedListWithKey
import pickle
import base64
import itertools

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
'''


class Board:
    # COLORS=['Y','B','P','G','C','R','M']
    _colors="red green yellow blue purple cyan magenta".split(' ')
    _colsize=len(_colors)

    _dirs = [
        [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]
    ]

    _straight_dirs=[]
    for dir in _dirs:
        if dir[0] and dir[1]:
            continue
        _straight_dirs.append(ddir(dir[0],dir[1]))

    _color_list = None
    _array = None
    _size = None
    _scrub_length=None
    _scrubs=None
    _tentative_scrubs=None
    _bg=None
    _assessment=None
    _axes= None

    iteration = None
    current_move = None

    _history=list()

    # axes: x - left, y - up, directions enumerated anti-clockwise

    def __init__(self, size=9, batch=5, colsize=None, scrub_length=5, axes=None):
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


    def get_segment_items(self,x,y,dx,dy,length=None):
        items=[]
        cx,cy=x,y
        i=0
        while self.valid(cx,cy):
            if length!=None and i >=length:
                break
            items.append([cx,cy, self.get_cell(cx,cy)])
            cx+=dx
            cy+=dy
            i+=1

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
            cx,cy,color=items[i]
            cand.poses[dpos(cx,cy)]=color
            if color!=None:
                cand.colors.setdefault(color, ddot(pos=list(),name=color,count=0,cand=cand
                                            , moves=SortedListWithKey(key=lambda move: -move.gain))
                                       )
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
            for i in range(0, length-self._scrub_length+1):
                cand,colors = self.candidate(x,y,dx,dy,items[i:i+self._scrub_length])
                # cand.index=len(self._cand_array)
                A.candidates.append(cand)
                for color in colors:
                    A.cand_colors.add(color)




    def find_best_move(self):
        if self.iteration==2:
            debug=1
        self.reset_assessment()
        self.candidates()
        self.color_assessment()
        best_move = self.pick_best_move()
        self.check_tentative_scrubs(best_move.pos_from,best_move.pos_to,best_move.color.name)
        return best_move

    def pick_best_move(self):
        A=self._assessment
        best_move=None
        for cand_color in A.cand_colors:
            for move in cand_color.moves:
                best_move= move
                break
            else:
                continue
            break
        else:
            return None

        return best_move

    def add_move_candidate(self, move, cand_color):
        A=self._assessment
        move_key = (move.pos_from, move.pos_to)
        if move_key in A.move_map:
            return

        cut_prob = self.assess_move(move)
        move.gain=cut_prob
        A.move_map[move_key] = move
        cand_color.moves.add(move)


    def color_assessment(self):
        counter=0
        for cand_color in self._assessment.cand_colors:
            self.assess(cand_color)
            counter+=1
            if counter>10:
                break


    def assess(self,cand_color):
        A=self._assessment
        self.assess_color_placement(cand_color)


    def assess_color_placement(self, cand_color):

        # A=self._assessment
        counter=0
        if cand_color.cand.free:
            for free_pos, metric in cand_color.cand.free.items():
                color_cells = self.get_cells_by_color(cand_color.name)
                for pos in color_cells:
                    if pos in cand_color.cand.poses:
                        continue
                    move = ddot(pos_from=pos, pos_to=free_pos, color=cand_color)
                    self.add_move_candidate(move, cand_color)
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
        # ca = self._bg.assess_connection_wo_node(start_node, end_node, start_edges, max_cut=3)
        # cut_prob = ca.cut_probability()
        lc = self._bg.fake_assess_connection_wo_node(start_node, end_node, start_edges, max_cut=3)
        cut_prob=0
        if lc == 0:
            cut_prob==1000
        else:
            cut_prob = 1/lc
        print ("%s to %s: %f" % (start_node, end_node, cut_prob))
        return cut_prob

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
                for j in range(max(i-self._scrub_length+1,0), min(i, length-self._scrub_length)+1):
                    cand[color][j][i-j]=(cx,cy)
            if len(cand):
                all_candidates.append([x,y,dx,dy,length,cand])
        return all_candidates

    def next_move(self):
        history_item = dict(board=dict(move=list(), remove=list(), new=list()), random_state=None)
        scrubbed=False
        if self.current_move:
            history_item['board']['move'].append((self.current_move.pos_from
                                        , self.current_move.pos_to
                                        , self.current_move.color.name))
            self.make_move(self.current_move)
            history_item['board']['remove']=self._scrubs
            if self._scrubs:
                scrubbed=True
            self.scrub_cells()
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
        self._bg.update_graph(self)
        self.current_move = self.find_best_move()
        return self.picked

    def history_post_load(self):
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

    def get_scrubs_XY(self,pos:dpos):
        x,y=tuple(pos)
        scrubs = []
        color = self._array[x][y]
        if color == None:
            return scrubs
        for dir in self._dirs[:4]:
            (dx, dy) = dir
            fw = self.get_segment_items(x, y, dx, dy)
            bw = self.get_segment_items(x, y, -dx, -dy)

            fw_i=next((i for i, x in enumerate(fw) if x[2]!=color), len(fw))
            bw_i=next((i for i, x in enumerate(bw) if x[2]!=color), len(bw))

            if fw_i+bw_i-1 < self._scrub_length:
                continue

            scrub = [ (v[0],v[1]) for v in itertools.chain(reversed(bw[1:bw_i]), fw[0:fw_i])]
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

        self.free_cell(pos_from,color)
        self.fill_cell(pos_to, color)
        self._tentative_scrubs = self.get_scrubs_XY(pos_to)
        self.free_cell(pos_to, color)
        self.fill_cell(pos_from, color)

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
                    self.free_cell(dpos(item[0], item[1]), self.get_cell(item[0],item[1]))
                    scrubbed[item]=1
        self._scrubs=None
        # self.update_buffer()


    def get_history(self):
        return self._history

    def set_history(self, history, iteration=None):
        for item in history:
            for i,move  in enumerate(item['board']['move']):
                _move=(dpos.fromdict(move[0]), dpos.fromdict(move[1]), move[2])
                item['board']['move'][i]=_move
            _scrubs=[]
            for i,scrub  in enumerate(item['board']['remove']):
                _scrub=[]
                for pos in scrub:
                    _scrub.append(dpos(pos[0], pos[1]))
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
