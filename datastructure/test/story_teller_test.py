from datastructure.story_teller import story_teller

import unittest
from unittest.mock import patch
from io import StringIO

class TestStory(unittest.TestCase):

    def test_story_teller(self):
        arr = [4,3,1,2]
        res = story_teller.minimumSwaps(arr)
        self.assertEqual(res, 3)

        arr = [5,1,2,3,4]
        res = story_teller.minimumBribes(arr)
        self.assertEqual(res, 4)

        # freqQuery: 1 is insert, 2 is delete, 3 is query, which has n frequency
        arr = [[1, 5], [1, 6], [3, 2], [1, 10], [1, 10], [1, 6], [2, 5], [3, 2]]
        res = story_teller.freqQuery(arr)
        self.assertEqual(res, [0, 1])

    def test_biggest_bst(self):
        newNode = story_teller.newNode
        largestBST = story_teller.largestBST

        self.assertEqual(largestBST(None), 0)

        root = newNode(10)
        self.assertEqual(largestBST(root), 1)

        root = newNode(20)
        root.left = newNode(10)
        root.right = newNode(30)
        self.assertEqual(largestBST(root), 3)

        root = newNode(50)
        root.left = newNode(10)
        root.right = newNode(60)
        root.left.left = newNode(5)
        root.left.right = newNode(20)
        root.right.left = newNode(55)
        root.right.left.left = newNode(45)
        root.right.right = newNode(70)
        root.right.right.left = newNode(65)
        root.right.right.right = newNode(80)
        self.assertEqual(largestBST(root), 6)

    @patch('sys.stdout', new_callable=StringIO)
    def test_printArray(self, mock_stdout):
        vInt = [1, 2, "a"]
        res = story_teller.printArray(vInt)
        output = mock_stdout.getvalue().strip().split('\n')
        self.assertEqual(output, ["1", "2", "a"])
        self.assertEqual(res, "a")

    @patch('sys.stdout', new_callable=StringIO)
    def test_two_sum_hashing(self, mock_stdout):
        num_arr = [4, 5, 5, 1, 8]
        pair_sum = 9
        story_teller.twoSumHashing(num_arr, pair_sum)
        output = mock_stdout.getvalue().strip()
        self.assertIn("Pair with sum 9 is: ( 5 , 4 )", output)
        self.assertIn("Pair with sum 9 is: ( 8 , 1 )", output)



if __name__ == '__main__':

    unittest.main()
