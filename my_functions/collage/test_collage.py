import unittest
from my_functions.collage.collage import topN

class TestCollageTopN(unittest.TestCase):
    def test_topN_typical(self):
        arr = {'a': 5, 'b': 2, 'c': 8, 'd': 1}
        result = topN(arr, 2)
        # Should return top 2 items sorted by value ascending
        self.assertEqual(result, [('d', 1), ('b', 2)])

    def test_topN_empty(self):
        arr = {}
        result = topN(arr, 2)
        self.assertEqual(result, [])

    def test_topN_zero_N(self):
        arr = {'a': 5, 'b': 2}
        result = topN(arr, 0)
        self.assertEqual(result, [])

    def test_topN_larger_N_than_dict(self):
        arr = {'a': 5, 'b': 2}
        result = topN(arr, 5)
        self.assertEqual(result, [('b', 2), ('a', 5)])

    def test_topN_duplicates(self):
        arr = {'a': 5, 'b': 2, 'c': 2, 'd': 5}
        result = topN(arr, 4)
        # 'b' and 'c' could be in any order but they will be before 'a' and 'd'
        self.assertEqual(len(result), 4)
        self.assertIn(result[0], [('b', 2), ('c', 2)])
        self.assertIn(result[1], [('b', 2), ('c', 2)])
        self.assertIn(result[2], [('a', 5), ('d', 5)])
        self.assertIn(result[3], [('a', 5), ('d', 5)])

if __name__ == '__main__':
    unittest.main()
