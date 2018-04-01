import unittest
from pprint import pprint
from board.board import Board
import random
import timeit
from board.utils import ddot,dpos,ddir,sign




class test_4_cutoff(unittest.TestCase):
    text_zero = '''
            . . . . . . . . .
            . . . . . . . . .
            . . . . . . . . .
            . . . . . . . . .
            . . . . . . . . .
            . . . . . . . . .
            . . . . . . . . .
            . . . . . . . . .
            . . . . . . . . .
            '''
    text0 = '''
            . y . . . r . . .
            r . . . . . . . p
            . y . p . . y y m
            c . r . . . . p .
            g b . p . . m . r
            . m . . . b . . .
            . . y . . . . c .
            . . . g . b . . .
            . . . . . g r . .
            '''

    text1 = '''
        . y . . . r . . .
        r . . . . . . . .
        . y . p . . y y .
        c . r . . . . p .
        g b . p . . . . .
        . m . . . b . . .
        . . y . . . . c .
        . . . g . b . . .
        . . . . . g r . .
        '''

    def setUp(self, text=None):
        random.seed(0)
        self.board = Board(size=9, batch=5, colsize=None, scrub_length=5, axes=None, logfile=None
                      , drawing_callbacks=dict())

        if text:
            self.board.init_test_graph(text)

    def test_cutoff0(self):
        self.board.init_test_graph(self.text0)
        print(self.board.dumps_test_graph())
        return

    def test_cutoff1(self):
        b=self.board
        b.init_test_graph(self.text0)
        print(b.dumps_test_graph())
        cs= b.cutoff(dpos(4,5))
        return

    def test_cutoff2(self):
        b = self.board
        b.init_test_graph(self.text1)
        print(b.dumps_test_graph())
        cs = b.cutoff(dpos(4, 5))
        return

    def test_cut_prob2(self):
        b = self.board
        b.init_test_graph(self.text1)
        print(b.dumps_test_graph())
        cs = b.cutoff(dpos(4, 5))
        return

    def test_cut_prob3(self):
        b = self.board
        total_disc_size, total_possible_disc_size = b.check_disc(dpos(3,3))
        # print(b.dumps_test_graph())
        # cs = b.cutoff(dpos(4, 5))
        return


    def bench0_setup(self):
        self.setUp(t.text0)
        b = self.board
        self.start = dpos(4,6)
        self.block = dpos(4,5)
        self.end = dpos(4,4)
        b.fill_cell(self.block, 'magenta')
        b.update_graph()

    def bench1_setup(self):
        self.setUp(t.text1)
        b = self.board
        self.start = dpos(4,6)
        self.block = dpos(4,5)
        self.end = dpos(4,4)
        # b.fill_cell(self.block, 'magenta')
        b.update_graph()

    def bench2_setup(self):
        self.setUp(t.text_zero)
        b = self.board
        self.start = dpos(4, 6)
        self.block = dpos(4, 5)
        self.end = dpos(4, 4)
        b.fill_cell(self.block, 'magenta')
        b.update_graph()

    def bench3_setup(self):
        self.setUp(t.text0)
        b = self.board
        self.block = dpos(4,4)
        # b.fill_cell(self.block, 'magenta')
        b.update_graph()


    def bench1(self):
        def test_ca():
            self.board._bg.assess_connection_wo_node(tuple(self.start), tuple(self.end), max_cut=3)

        def test_cs():
            cs = self.board.cutoff(dpos(4, 5))

        r1 = timeit.repeat(test_ca, repeat=1, number=500)
        r2 = timeit.repeat(test_cs,repeat=1,number=500)
        print(r1,r2)

    def bench2(self):
        def test_cs():
            cs = self.board.cutoff(dpos(4, 5))
            return cs

        def test_cs2():
            cs = self.board.cutoff2(dpos(4, 5))
            return cs

        cs1=test_cs()
        cs2=test_cs2()
        print ("cs1:")
        for cs in cs1.cutsets:
            print([tuple(node) for node in cs])
        print("cs1:")

        for cs in cs2.cutsets:
            print([tuple(node) for node in cs])

        print(cs1.cutsets==cs2.cutsets)

        r1 = timeit.repeat(test_cs, repeat=1, number=100)
        r2 = timeit.repeat(test_cs2,repeat=1,number=100)
        print(r1,r2)


if __name__=='__main__':
    t = test_4_cutoff()
    t.bench3_setup()
    t.test_cut_prob3()


