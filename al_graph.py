import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import random,math,copy
from random import randrange,randint
from matplotlib.widgets import Button

from board import Board

def main():
    # import pdb; pdb.set_trace()
    board = Board()
    board.init(9,5,None,5)
    random.seed(0)
    free_cells =0 

    # def make_graph():
    #     a = board.array
    #     G = nx.grid_graph([board._size,board._size])
    #     attrs={}
    #     for i in range(board._size):
    #         for j in range(board._size):
    #             data={ 'pos': (i,j), 'occupied': a[i][j] } 
    #             attrs[(i,j)]=data
    #     nx.set_node_attributes(G,attrs)
    #     return G   

    def start():
        # G=make_graph()
        board._bg.draw(main_ax)

    
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

    def on_click(event):
        # if event.click:
           # ax.plot((event.xdata, event.xdata), (mean-standardDeviation, mean+standardDeviation), 'r-')
        # import pdb;pdb.set_trace() # ddd
        picked= next_move()
        # G=make_graph()
        board._bg.draw(main_ax)

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
