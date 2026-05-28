import unittest
from Temp import solution

class TestTemp(unittest.TestCase):
    def test_empty_grid(self):
        self.assertEqual(solution([]), 0)

    def test_empty_row_grid(self):
        self.assertEqual(solution([[]]), 0)

    def test_single_element_grid(self):
        self.assertEqual(solution([[5]]), -float('inf'))

    def test_example_grid(self):
        grid = [
            [10, 3, 6, 4],
            [5, 11, 8, 2],
            [7, 8, 15, 2],
            [4, 6, 5, 13]
        ]
        self.assertEqual(solution(grid), 12)  # 15 - 3 = 12, going right and down from 3

    def test_all_negative_grid(self):
        grid = [
            [-5, -2, -9],
            [-3, -8, -1],
            [-4, -7, -6]
        ]
        self.assertEqual(solution(grid), 8)  # -1 - (-9) = 8

    def test_increasing_grid(self):
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        self.assertEqual(solution(grid), 8)  # 9 - 1 = 8

    def test_decreasing_grid(self):
        grid = [
            [9, 8, 7],
            [6, 5, 4],
            [3, 2, 1]
        ]
        self.assertEqual(solution(grid), -1)  # going down or right must decrease by at least 1, max is -1
        # wait, is the max -1?
        # 9 -> 8 (-1)
        # 8 -> 7 (-1)
        # 6 -> 5 (-1)
        # etc.
        # So yes, -1.

if __name__ == '__main__':
    unittest.main()
