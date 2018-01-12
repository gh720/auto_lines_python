import unittest
from pprint import pprint
from board import Board

class  TestBoard1(unittest.TestCase):
    def setUp(self):
        self.board=Board()
        self.board.init(9,5,None,5)

    def test_lines(self):
        lines =self.board.get_lines()
        pprint(lines)

if __name__=='__main__':
    unittest.main()
