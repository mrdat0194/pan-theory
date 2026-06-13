import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from submit import LongestIncreasingSubsequenceLength

class TestLongestIncreasingSubsequenceLength(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(LongestIncreasingSubsequenceLength([]), 0)

    def test_single_element(self):
        self.assertEqual(LongestIncreasingSubsequenceLength([5]), 1)

    def test_all_increasing(self):
        self.assertEqual(LongestIncreasingSubsequenceLength([1, 2, 3, 4, 5]), 5)

    def test_all_decreasing(self):
        self.assertEqual(LongestIncreasingSubsequenceLength([5, 4, 3, 2, 1]), 1)

    def test_mixed(self):
        v = [2, 5, 3, 7, 11, 8, 10, 13, 6]
        self.assertEqual(LongestIncreasingSubsequenceLength(v), 6)

    def test_duplicates(self):
        self.assertEqual(LongestIncreasingSubsequenceLength([2, 2, 2, 2]), 1)

    def test_negative_numbers(self):
        self.assertEqual(LongestIncreasingSubsequenceLength([-5, -1, -3, -2, 0]), 4)

if __name__ == '__main__':
    unittest.main()
