import unittest
from pprint import pprint
from board import Board
import random

@unittest.skip('')
class  test_board_1_lines_gen(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        self.board=Board()
        self.board.init(9,5,None,5)
        
    def test_lines(self):
        for i in range(5): 
            self.board.next_move()

        lines =self.board.get_lines()
        pprint(lines)

# @unittest.skip('')
class test_board_2_candidates(unittest.TestCase):
    def setUp(self):
        random.seed(2)
        self.board=Board()
        self.board.init(9,1,None,5)

    def test_candidates(self):
        for i in range(10): 
            self.board.next_move()
        # import pdb;pdb.set_trace()
        cand=self.board.candidates()
        pprint(cand)

@unittest.skip('')
class test_ray_hit(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        self.board=Board()
    def test_ray_hit(self):
        sides=[[0,None], [None,0], [8,None], [None,8]]
        dirs =self.board._dirs
        x,y=2,3
        for dx,dy in dirs:
            for side in sides:
                hit = self.board.ray_hit(x,y,dx,dy,side)
                print(f"x={x} y={y} dx={dx} dy={dy} side={side} hit={hit}\n")

        # self.board.ray_hit()

# class test_path


if __name__=='__main__':
    unittest.main()


