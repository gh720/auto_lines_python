import copy 
from random import randrange,randint
import collections
from board_graph import Board_graph
from attrdict import AttrDict



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
    _assessment=None


    # axes: x - left, y - up, directions enumerated anti-clockwise
    _dirs = [
            [1,0], [1,1], [0,1], [-1,1], [-1,0], [-1,-1], [0,-1], [1,-1]
        ]

    def init(self, size=9, batch=5, colsize=None, _scrap_length=5):
        if colsize==None:
            colsize=len(self._colors)
        self._size=size
        self._batch=batch
        self._scrap_length=_scrap_length
        self._colsize=colsize
        self._sides=[[0,None], [None,0], [self._size-1,None], [None,self._size-1]]
        self._array = [
            [None for i in range(0,self._size)]
                for j in range(0,self._size)
        ]
        self._color_list=collections.defaultdict(dict)
        self.update_buffer()
        self.reset_assessment()
        self._bg.update_graph(self)
        # self.prepare_view()

    def get_array(self):
        return self._array

    def get_size(self):
        return self._size

    def reset_assessment(self):
        self._assessment=AttrDict(
            move_in=None, move_out=None, move_in_out=None
        )

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
        colors = dict()
        free=dict()
        total_colors=0
        for i in range(len(items)):
            color,cx,cy=items[i]
            if color!=None:
                colors.setdefault(color,{'pos':(cx,cy),'count':0})['count']+=1
                total_colors+=1
            else:
                free[(cx,cy)]={'lc':self._bg.metrics['lc'][(cx,cy)]}
        return colors,free,total_colors


    def candidates(self):
        lines = self.get_chords()
        best_cand=None
        for line in lines:
            x,y,dx,dy,length,items=line
            for i in range(0, length-self._scrap_length+1):
                cand= self.candidate(x,y,dx,dy,items[i:i+self._scrap_length])
                self.assess(cand)

    # def assess_placement(self,color,x,y):
    #     positions = self._color_list[color]
    #     for cx,cy in positions:
    #         self._bg.find_path(())



    def assess(self,cand):
        colors,free,total_colors = cand
        A=self.assessment

        for color,data in colors.items():
            move_out=(total_colors-data['count'])
            move_in = free + move_out
            move_in_out=move_out+move_in
            if A.move_in_out ==None:
                pass # TODO: impl
            if A.move_in_out < move_in_out:
                continue
            if A.move_in_out==move_in_out:
                self.assess_placement()



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
        picked= self.get_random_free_cells()
        self.place(picked)
        self._bg.update_graph(self)
        self.last_balls= self.picked_balls(picked)
        return picked

    def place(self,picked):
        for item in picked:
            (x,y,color)=item
            self._array[x][y]=color
            self._color_list[color][(x,y)]=1
            # self.color_lists[color]
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

    def cell(self,i,j):
        return self._array[i][j]

    def get_free_cells(self):
        free=[]
        for i in range(0,self._size):
            for j in range(0,self._size):
                if (self.cell(i,j)==None):
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

    def check_scraps(self):
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

