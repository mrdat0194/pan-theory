import unittest
from Temp import solution

class TestTempSolution(unittest.TestCase):

    def test_example_normal_grid(self):
        # Example grid from the script
        grid = [
            [10, 3, 6, 4], 
            [5, 11, 8, 2], 
            [7, 8, 15, 2], 
            [4, 6, 5, 13]
        ]
        self.assertEqual(solution(grid), 12)

    def test_empty_grid(self):
        self.assertEqual(solution([]), 0)
        
    def test_empty_row_grid(self):
        self.assertEqual(solution([[]]), 0)

    def test_one_by_one_grid(self):
        # Only 1 element, no valid end value exists
        grid = [[5]]
        self.assertEqual(solution(grid), -float('inf'))
        
    def test_grid_decreasing_values(self):
        # In a completely decreasing grid, moving right or down always decreases the value
        grid = [
            [20, 15],
            [15, 10]
        ]
        self.assertEqual(solution(grid), -5)
        
    def test_grid_negative_values(self):
        grid = [
            [-10, -5],
            [-15, -2]
        ]
        # max_val_suffix table:
        # [-2, -2]
        # [-2, -2]
        # Max score is from -15 to -2, which is 13
        # No, let's trace:
        # -10 to -2 = 8
        # -15 to -2 = 13
        self.assertEqual(solution(grid), 13)

    def test_grid_linear(self):
        grid = [[1, 2, 3, 4]]
        self.assertEqual(solution(grid), 3) # 4 - 1
        
        grid2 = [[1], [2], [3], [4]]
        self.assertEqual(solution(grid2), 3) # 4 - 1

if __name__ == '__main__':
    unittest.main()
