import subprocess

LEFT=-1
RIGHT=1
DIR_DICT = { (0,1): 0, (1,0):1, (0,-1):2, (-1,0):3}
DIRS = [(0,1),(1,0),(0,-1),(-1,0)]


def direction(node_a,node_b)
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

