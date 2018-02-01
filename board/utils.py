# import subprocess

LEFT=-1
RIGHT=1
DIR_DICT = { (0,1): 0, (1,0):1, (0,-1):2, (-1,0):3}
DIRS = [(0,1),(1,0),(0,-1),(-1,0)]

def listget(i,lst):
    if not lst or i<0 or i>=len(lst):
        return None
    return lst[i]

def direction(node_a,node_b):
    return DIR_DICT[(node_b[0]-node_a[0],node_b[1]-node_a[1])]
    

# count of turns clock-wise to get to dir2 from dir1
def turns(dir1,dir2):
    turns = (dir2+len(DIRS)-dir1)%len(DIRS)
    return turns


def next_in_loop(array,i):
    if len(array)==0:
        return None
    if len(array)==1:
        return 0
    ni = (i+1)%len(array)
    return array[ni],ni

def prev_in_loop(array,i):
    if len(array)==0:
        return None
    if len(array)==1:
        return 0
    ni = (i-1+len(array))%len(array)
    return array[ni],ni

def next_i_in_loop(array,i):
    if len(array)==0:
        return None
    if len(array)==1:
        return 0
    return (i+1)%len(array)

def prev_i_in_loop(array,i):
    if len(array)==0:
        return None
    if len(array)==1:
        return 0
    return (i-1+len(array))%len(array)


def clear():
    subprocess.check_call('cls', shell=True)

class _Getch:
    """
    Gets a single character from standard input.  Does not echo to the
    screen.
    """
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self):
        return self.impl()


class _GetchUnix:
    def __call__(self):
        import sys
        import tty
        import termios

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt  # NOQA

    def __call__(self):
        import msvcrt

        return msvcrt.getch()


getch = _Getch()

class ddot(dict):
    def __init__(self, *args, **kwargs):
        super(ddot, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @classmethod
    def recursive_walk(cls, d):
        self = cls(d)
        for key, value in self.items():
            if type(value) is dict:
                self[key] = cls.recursive_walk(value)
        return self

class dpos(ddot):
    def __init__(self, x, y):
        # assert x>=0 and y>=0
        super().__init__(x=x,y=y)

    # def __setattr__(self,key,value):
    #     if key == 'x' and value<0:
    #         debug=1
    #     super(ddot,self).__setattr__(key,value)

    def copy(self):
        return dpos(self.x,self.y)

    def __hash__(self):
        return (self.x,self.y).__hash__()

    def __iter__(self):
        yield self.x
        yield self.y

    @classmethod
    def fromdict(cls, dict):
        return cls(dict['x'], dict['y'])


class ddir(ddot):
    def __init__(self, x, y):
        super().__init__(dx=x,dy=y)

    def __hash__(self):
        return (self.dx,self.dy).__hash__()

    def copy(self):
        return ddir(self.dx,self.dy)

    def __iter__(self):
        yield self.dx
        yield self.dy

    @classmethod
    def fromdict(cls, dict):
        return cls(dict['dx'], dict['dy'])

