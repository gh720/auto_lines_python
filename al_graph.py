import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import random,math,copy
from random import randrange,randint
from matplotlib.widgets import Button
import argparse
import json


from board.board import Board

def main():
    # import pdb; pdb.set_trace()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--history", help="position history file")
    parser.add_argument("-i", "--iteration", help="position history file")
    parser.add_argument("-m", "--move", help="start from this position")
    parser.add_argument("-l", "--log", help="log computation details")
    parser.add_argument("-mf", "--max_free_moves", help="max free moves to assess at each recursion")
    parser.add_argument("-mo", "--max_obstacle_moves", help="max obstacle removals to assess at each level of recursion")
    parser.add_argument("-dh", "--debug_hqueue", help="diagnostics level for hqueue")

    nargs, args = parser.parse_known_args()

    board = None
    random.seed(0)
    history_file=nargs.history or None
    default_history_file= 'al.history'
    iteration=nargs.iteration or None
    free_cells =0
    logfile = nargs.log or None

    stages = ('assessment', 'after_throw', 'move_found', 'over')
    stages_to_show=('assessment', 'after_throw', 'move_found', 'over')

    game_over=False

    def next_move():
        nonlocal board
        # scraps =board.check_scraps()
        # if (len(scraps) >0):
        #     board.draw_scraps()
        #     # board.prepare_view()
        #     # draw()
        #     board.scrap_cells()
        #     return

        free_cells = board.get_free_cells()
        if len(free_cells)==0:
            # message("game over")
            return

        new_balls = board.next_move()

    # def draw():
    #     board.draw_move()



    def drawing_callback(stage):
        if stage in stages:
            board.log("show: stage %s, I:%s" % (stage, ['off','on'][plt.isinteractive()]))
            plt.pause(0.001)
            if stage=='over':
                game_over=True
            # plt.show(block=False)


    def start():
        nonlocal board,nargs
        # G=make_graph()
        board= Board(size=9,batch=5,colsize=None,scrub_length=5,axes=main_ax, logfile=logfile
                     , drawing_callbacks= { stage: (drawing_callback if stage in stages_to_show else None)
                                            for stage in stages }
                     )
        if nargs.debug_hqueue!=None:
            board.set_debug_levels({'hqueue': int(nargs.debug_hqueue)})
        # board.draw(show=False)
        if nargs.max_free_moves!=None:
            board.max_free_moves=int(nargs.max_free_moves)
        if nargs.max_obstacle_moves != None:
            board.max_obstacle_moves = int(nargs.max_obstacle_moves)
        if history_file:
            load_history()
        else:
            picked=next_move()
        # draw()

    def save_history():
        nonlocal board
        history = board.get_history()
        _history_file=history_file or default_history_file
        with open(_history_file, 'w') as fh:
            fh.write(json.dumps(history))

    def load_history():
        nonlocal board
        try:
            with open(history_file, 'r') as fh:
                jstr = fh.read()
                history = json.loads(jstr)
                board.set_history(history,iteration)
        except FileNotFoundError:
            if iteration!=None:
                raise


    def on_click(event):
        # if event.click:
           # ax.plot((event.xdata, event.xdata), (mean-standardDeviation, mean+standardDeviation), 'r-')
        # import pdb;pdb.set_trace() # ddd
        if not game_over:
            next_move()
        # G=make_graph()
        # draw()

    def on_keypress(event):
        if event.key=='s':
            save_history()

    def _exit():
        exit()

    figure = plt.figure(1,figsize=(6,6))
    main_ax = plt.subplot(111)
    thismanager = plt.get_current_fig_manager()
    thismanager.window.wm_geometry("+660+0")

    # main_ax= plt.gca()

    plt.connect('button_press_event', on_click)

    axcut = plt.axes([0.9, 0.5, 0.1, 0.075])
    figure.tight_layout()
    bcut = Button(axcut, 'YES', color='red', hovercolor='green')
    bcut.on_clicked(_exit)

    plt.sca(main_ax)

    mpl.rcParams['keymap.save'].remove('s')
    mpl.rcParams['keymap.quit'].remove('q')

    plt.connect('key_press_event', on_keypress)

    # plt.show()

    start()

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()
