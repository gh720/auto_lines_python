import copy 
from random import randrange,randint

class Board:
    # COLORS=['Y','B','P','G','C','R','M']
    _colors="red green yellow blue purple cyan magenta".split(' ')
    _colsize=len(_colors)
    _scraps=[]

    # axes: x - left, y - up, directions enumerated anti-clockwise
    dirs = [
            [1,0], [1,1], [0,1], [-1,1], [-1,0], [-1,-1], [0,-1], [1,-1]
        ]

    def init(self, size=9, batch=5, colsize=None, _scrap_length=5):
        if colsize==None:
            colsize=len(self._colors)
        self._size=size
        self._batch=batch
        self._scrap_length=_scrap_length
        self._colsize=colsize
        self.array = [
            [None for i in range(0,self._size)]
                for j in range(0,self._size)
        ]
        self.update_buffer()
        # self.prepare_view()

    
    # def to_graph(self):
    #     nodes=[]
    #     a= self.array
    #     for i in range(self.size):
    #         for j in range(self.size):

    # class line:
    #     length=0
    #     items=[]

    def get_cell(self,x,y):
        return self.array[x][y]

    def set_cell(self,x,y,value):
        self.array[x][y]=value

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


    def get_lines(self):
        lines=[]
        for k in range(0,4):
            dx,dy=self.dirs[k]
            starts=self.get_starts(dx,dy)

            for x,y,length in starts:
                items=[]
                cx,cy=x,y
                for l in range(length):
                    items.append(self.get_cell(cx,cy))
                    cx+=dx
                    cy+=dy
                lines.append([x,y,k,length,items])
        return lines


    def candidates(self,color):
        # hash of cand.lines by cell pos
        # cand.lines have obstacle info
        cand=collections.defaultdict(dict)
        lines = self.get_lines()
        all_candidates = []
        for line in lines:
            x,y,dir,length,items=line
            for i in range(length):
                color=items[i]
                for j in range(max(i-4,0),min(i+4,length)+1):
                    cand[color][j][i]=1
            all_candidates.append([x,y,dir,length,cand])
        return all_candidates

    def picked_balls(self,picked):
        msg=",".join(["%s[%s,%s]" % (picked[i][2],picked[i][0],picked[i][1]) for i in range(0,len(picked))])
        return msg

    def next_move(self):
        picked= self.get_random_free_cells()
        self.place(picked)
        self.last_balls= self.picked_balls(picked)
        return picked

    def place(self,picked):
        for item in picked:
            (x,y,color)=item
            self.array[x][y]=color
            # self.color_lists[color]
        self.update_buffer()

    def update_buffer(self):
        self._buffer=copy.deepcopy(self.array)

    # def prepare_view(self):
    #     rows = []
    #     for i in range(0,self._size):
    #         rows.append("  ".join([ '.' if x ==' ' else x for x in self._buffer[i]]))
    #     self.view = "\n".join(rows)

    def cell(self,i,j):
        return self.array[i][j]

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
        # dirs = [
        #     [1,0], [1,1], [0,1], [-1,1]
        # ]
        scraps = []
        color = self.array[x][y]
        if color == None:
            return scraps
        for dir in self.dirs[:4]:
            scrap = [[x,y]]
            (sx,sy)=dir
            (cx,cy)=(x,y)
            while True:
                cx+=sx
                cy+=sy
                if not self.valid(cx,cy):
                    break
                if self.array[cx][cy]==color:
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
                self.array[item[0]][item[1]]=None
        # self.update_buffer()