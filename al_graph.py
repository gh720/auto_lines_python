import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import random,math,copy
from random import randrange,randint
from matplotlib.widgets import Button
import collections

from board import Board


gna=nx.get_node_attributes; sna=nx.set_node_attributes

# def _init():
#     random.seed(0)
#     size=9
#     dice_times=20



def find_free(nodeset,x,y):
    global size; cx=x; cy=y
    if (x,y) in nodeset: return (x,y)
    while True:
        cy+=1
        if (cy>=size): 
            cy=0;cx+=1
            if (cx>=size): cx=0
        if cx==x and cy==y:
            return None
        if (cx,cy) in nodeset:
            return (cx,cy)            
def throw_dice(nodeset,n):
    global size
    thrown=dict()
    n=min(len(nodeset), n)
    nodeset_copy=set.copy(nodeset)
    for i in range(n):
        _x= randrange(size); _y=randrange(size)
        free_node=find_free(nodeset_copy,_x,_y)
        assert free_node!=None
        if free_node in thrown:
            display(thrown)
            display((_x,_y))
            display(free_node)
        assert free_node not in thrown
        nodeset_copy.remove(free_node)
        thrown[free_node]=colors[randrange(len(colors))]
    return thrown

def main():
    # import pdb; pdb.set_trace()
    global G
    board = Board()
    board.init(9,5,None,5)
    free_cells =0 
    G=None

    def make_graph():
        a = board.array
        G = nx.grid_graph([board._size,board._size])
        attrs={}
        for i in range(board._size):
            for j in range(board._size):
                data={ 'pos': (i,j), 'occupied': a[i][j] } 
                attrs[(i,j)]=data
        nx.set_node_attributes(G,attrs)
        return G   

    def start():
        global G
        G=make_graph()
        draw(main_ax)

    def draw(axes):
        global G
        if G==None:
            return

        plt.sca(axes)
        plt.cla()

        OG = G.subgraph([ x for x,data in G.nodes.items() if data['occupied']!=None] ).copy()
        nx.set_node_attributes(OG,gna(OG,'occupied'),'colors')
        FG=G.copy(); FG.remove_nodes_from(OG)
        sna(FG,nx.load_centrality(FG),'lc_metric')
        ap=nx.articulation_points(FG)
        APG=FG.subgraph(ap).copy()

        text={ (x,y):(x,y+.3) for x,y in gna(G,'pos') };

        plt.figure(1,figsize=(8,8))
        nx.draw_networkx(FG,pos=gna(FG,'pos')
            , with_labels=False, node_shape='s',node_size=100, cmap=plt.get_cmap('plasma')
            , node_color=[gna(FG,'lc_metric').get(x,0)for x in list(FG)] )
        nx.draw_networkx(OG,pos=gna(OG,'pos'), with_labels=False, node_shape='o',node_size=500
            , node_color=[ mpl.colors.cnames[y] for x,y in gna(OG,'colors').items()] )
        # nx.draw_networkx_edges(TG, gna(TG,'pos'), sp_edges, width=3.0, edge_color='r', arrow_style='>')
        nx.draw_networkx_nodes(APG, pos=gna(FG,'pos'), node_shape='d', node_color='r', node_size=300)
        nx.draw_networkx_labels(G, pos=text, labels={ (x,y):str((x,y)) for x,y in gna(G,'pos') })
        plt.show()

    def next_move():
        scraps =board.check_scraps()
        if (len(scraps) >0):
            board.draw_scraps()
            # board.prepare_view()
            # draw()
            board.scrap_cells()
            return

        free_cells = board.get_free_cells()
        if len(free_cells)==0:
            message("game over")
            return

        new_balls = board.next_move()
        if len(new_balls)>0:
            # board.prepare_view()
            pass
            # draw()

    def _exit():
        exit()

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


    def on_click(event):
        global G
        # if event.click:
           # ax.plot((event.xdata, event.xdata), (mean-standardDeviation, mean+standardDeviation), 'r-')
        # import pdb;pdb.set_trace() # ddd
        picked= next_move()
        G=make_graph()
        draw(main_ax)

    ax = plt.subplot(111)
    main_ax= plt.gca()

    plt.connect('button_press_event', on_click)

    axcut = plt.axes([0.9, 0.0, 0.1, 0.075])
    bcut = Button(axcut, 'YES', color='red', hovercolor='green')
    bcut.on_clicked(_exit)
    
    # plt.show()

    start()

if __name__ == '__main__':
    main()
