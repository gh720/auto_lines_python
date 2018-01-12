from random import randint, randrange
from utils import clear

import sys, msvcrt, copy

class Board:
    COLORS=['Y','B','P','G','C','R','M']
    colsize=len(COLORS)
    scraps=[]

    def init(self, size=9, batch=5, colsize=None, scrap_length=5):
        if colsize==None:
            colsize=len(COLORS)
        self._size=size
        self._batch=batch
        self._scrap_length=scrap_length
        self.colsize=colsize
        self.array = [
            [' ' for i in range(0,self._size)]
                for j in range(0,self._size)
        ]
        self.update_buffer()
        self.prepare_view()

    
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
            self.array[item[0]][item[1]]=item[2]
        self.update_buffer()

    def update_buffer(self):
        self._buffer=copy.deepcopy(self.array)

    def prepare_view(self):
        rows = []
        for i in range(0,self._size):
            rows.append("  ".join([ '.' if x ==' ' else x for x in self._buffer[i]]))
        self.view = "\n".join(rows)

    def cell(self,i,j):
        return self.array[i][j]

    def get_free_cells(self):
        free=[]
        for i in range(0,self._size):
            for j in range(0,self._size):
                if (self.cell(i,j)==' '):
                    free.append([i,j,self.COLORS[randrange(self.colsize)]])
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
        dirs = [
            [1,0], [1,1], [0,1], [-1,1]
        ]
        scraps = []
        color = self.array[x][y]
        if color == ' ':
            return scraps
        for dir in dirs:
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
        self.prepare_view()
        self.scrap_cells()

    def scrap_cells(self,scraps=None):
        self.scraps = scraps if scraps else self.scraps
        for scrap in self.scraps:
            for item in scrap:
                self.array[item[0]][item[1]]=' '
        # self.update_buffer()


def main():
    # import pdb; pdb.set_trace()
    board = Board()
    board.init(9,5,3,5)

    
    free_cells =0 

    def message(message):
        sys.stdout.write("** {0} **\n".format(message))

    def draw():
        clear()
        sys.stdout.write('{0}\n\n'.format(board.view))
        sys.stdout.write('== {0} ==\n'.format(board.last_balls))

    def next_move():
        scraps =board.check_scraps()
        if (len(scraps) >0):
            # import pdb; pdb.set_trace()
            board.draw_scraps()
            board.prepare_view()
            draw()
            board.scrap_cells()
            return

        free_cells = board.get_free_cells()
        if len(free_cells)==0:
            message("game over")
            return

        # import pdb; pdb.set_trace()

        new_balls = board.next_move()
        if len(new_balls)>0:
            board.prepare_view()
            draw()

    KEYS = {
        # 'w': view.cursor_up,
        # 's': view.cursor_down,
        # 'a': view.cursor_left,
        # 'd': view.cursor_right,
        # ' ': move,
        # 'u': undo,
        # 'r': redo,
        b' ': next_move,
        b'\x1b': exit,
        b'q': exit,
    }

    while True:
        
        c =msvcrt.getch()
        try:
            sys.stdout.write("pressed key %s\n" % (str(c)))
            KEYS[c]()

        # except BoardError as be:
        #     err = be.message
        except KeyError as ke:
            sys.stderr.write("wrong key %s\n" % (str(ke)))
            pass

if __name__ == '__main__':
    main()
