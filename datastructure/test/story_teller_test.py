import sys
import unittest
from unittest.mock import patch
from io import StringIO

sys.setrecursionlimit(10000)

from datastructure.story_teller import (
    story_teller,
    ArrayProblems,
    Stock,
    FibonacciShowcase,
    DynamicProgramming,
    SearchAndCount,
    GeometryProblems,
    ClassicPuzzles,
    PythonPatterns,
    SnippetExamples,
    SubsetProblems,
    DifferenceArray,
    StockProfit,
    RiverSizes,
    ArrayUtils,
    GraphProblems,
    DisjointSet,
    KunduAndTree,
    SuperMaximumCostQuery,
    BSTShowcase,
    CameraCoverSolution,
    SynonymQueries,
)

class TestStory(unittest.TestCase):

    def test_story_teller(self):
        arr = [4, 3, 1, 2]
        res = story_teller.minimumSwaps(arr)
        self.assertEqual(res, 3)

        arr = [5, 1, 2, 3, 4]
        res = story_teller.minimumBribes(arr)
        self.assertEqual(res, 4)

        # freqQuery: 1 is insert, 2 is delete, 3 is query, which has n frequency
        arr = [[1, 5], [1, 6], [3, 2], [1, 10], [1, 10], [1, 6], [2, 5], [3, 2]]
        res = story_teller.freqQuery(arr)
        self.assertEqual(res, [0, 1])


    @patch('sys.stdout', new_callable=StringIO)
    def test_two_sum_hashing(self, mock_stdout):
        num_arr = [4, 5, 5, 1, 8]
        pair_sum = 9
        story_teller.twoSumHashing(num_arr, pair_sum)
        output = mock_stdout.getvalue().strip()
        self.assertIn("Pair with sum 9 is: ( 5 , 4 )", output)
        self.assertIn("Pair with sum 9 is: ( 8 , 1 )", output)

    def test_two_strings(self):
        self.assertEqual(story_teller.twoStrings("hello", "world"), "YES")
        self.assertEqual(story_teller.twoStrings("would", "xyz"), "NO")

    def test_sherlock_and_anagrams(self):
        self.assertEqual(story_teller.sherlockAndAnagrams("ifailuhkqq"), 3)
        self.assertEqual(story_teller.sherlockAndAnagrams("kkkk"), 10)

    def test_make_anagram(self):
        self.assertEqual(story_teller.makeAnagram("cde", "abc"), 4)

    def test_check_magazine_and_ransom_note(self):
        mag = ["give", "me", "one", "grand", "today", "night"]
        note = ["give", "one", "grand", "today"]
        self.assertTrue(story_teller.checkMagazine(mag, note))
        self.assertTrue(story_teller.ransom_note(mag, note))
        
        note_bad = ["give", "one", "grand", "today", "tomorrow"]
        self.assertFalse(story_teller.checkMagazine(mag, note_bad))
        self.assertFalse(story_teller.ransom_note(mag, note_bad))

    def test_sock_merchant(self):
        self.assertEqual(story_teller.sockMerchant(9, [10, 20, 20, 10, 10, 30, 50, 10, 20]), 3)

    def test_counting_valleys(self):
        self.assertEqual(story_teller.countingValleys(8, "UDDDUDUU"), 1)

    def test_pyramid_build(self):
        self.assertEqual(story_teller.pyramid_build(6, [1, 1, 3, 3, 2, 1]), [0, 1, 2, 3, 2, 1])

    def test_repeated_string(self):
        self.assertEqual(story_teller.repeatedString("aba", 10), 7)

    def test_count_triplets(self):
        self.assertEqual(story_teller.countTriplets([1, 5, 5, 25, 125], 5), 4)

    def test_forming_magic_square(self):
        self.assertEqual(story_teller.formingMagicSquare([4, 9, 2, 3, 5, 7, 8, 1, 5]), 1)

    def test_largest_bst_bt(self):
        BSTNode = BSTShowcase.BSTNode
        
        # Test empty tree
        self.assertEqual(BSTShowcase.largestBSTBT(None)[3], 0)

        # Test single node
        root = BSTNode(10)
        self.assertEqual(BSTShowcase.largestBSTBT(root)[3], 1)

        # Test valid BST
        root = BSTNode(20)
        root.left_child = BSTNode(10)
        root.right_child = BSTNode(30)
        self.assertEqual(BSTShowcase.largestBSTBT(root)[3], 3)

        # Test complex binary tree
        root = BSTNode(50)
        root.left_child = BSTNode(10)
        root.right_child = BSTNode(60)
        root.left_child.left_child = BSTNode(5)
        root.left_child.right_child = BSTNode(20)
        root.right_child.left_child = BSTNode(55)
        root.right_child.left_child.left_child = BSTNode(45)
        root.right_child.right_child = BSTNode(70)
        root.right_child.right_child.left_child = BSTNode(65)
        root.right_child.right_child.right_child = BSTNode(80)
        self.assertEqual(BSTShowcase.largestBSTBT(root)[3], 6)

    @patch('sys.stdout', new_callable=StringIO)
    def test_print_paths(self, mock_stdout):
        BSTNode = BSTShowcase.BSTNode
        root = BSTNode(10)
        root.left_child = BSTNode(8)
        root.right_child = BSTNode(2)
        BSTShowcase.printPaths(root)
        output = mock_stdout.getvalue().strip().split('\n')
        self.assertEqual(output, ["10 8", "10 2"])

    def test_goodness(self):
        self.assertEqual(story_teller.goodness("defGEhfX2"), 5)

    def test_roads_and_libraries(self):
        self.assertEqual(story_teller.roadsAndLibraries(3, 2, 1, [[1, 2], [3, 1], [2, 3]]), 4)

    def test_find_shortest(self):
        self.assertEqual(story_teller.findShortest(5, [1, 1, 2, 3], [2, 3, 4, 5], [1, 2, 3, 3, 2], 2), 3)

    def test_closest_to_zero(self):
        self.assertEqual(story_teller.closestToZero([-2, 1, 3, 5]), 1)

    def test_shuffle_deck(self):
        deck = story_teller.shuffleDeck()
        self.assertEqual(len(deck), 52)
        self.assertTrue(hasattr(deck[0], 'suit'))
        self.assertTrue(hasattr(deck[0], 'rank'))

    def test_make_cartree(self):
        from datastructure.story_teller import make_cartree
        cartree = make_cartree([4, 3, 7, 2, 6, 1, 9], None, None)
        self.assertEqual(
            str(cartree),
            "1=[l->2=[l->3=[l->4=[l->None, r->None], r->7=[l->None, r->None]], r->6=[l->None, r->None]], r->9=[l->None, r->None]]"
        )

    @patch('sys.stdout', new_callable=StringIO)
    def test_player_score_billboard(self, mock_stdout):
        from datastructure.story_teller import Player
        data = [
            Player("amy", 100),
            Player("david", 100),
            Player("heraldo", 50),
            Player("aakansha", 75),
            Player("aleksa", 150)
        ]
        Player.score_billboard(data)
        output = mock_stdout.getvalue().strip().split('\n')
        expected = [
            "aleksa 150",
            "amy 100",
            "david 100",
            "aakansha 75",
            "heraldo 50"
        ]
        self.assertEqual(output, expected)

    def test_compare_triplets(self):
        from datastructure.story_teller import Player
        self.assertEqual(Player.compareTriplets([17, 28, 30], [99, 16, 8]), [2, 1])

    def test_climbing_leaderboard(self):
        from datastructure.story_teller import Player
        scores = [100, 90, 90, 80, 75, 60]
        alice = [50, 65, 77, 90, 102]
        res = Player.climbingLeaderboard(scores, alice)
        self.assertEqual(res, [6, 5, 4, 2, 1])

    def test_next_permutation(self):
        from datastructure.story_teller import Solution
        arr = [0, 1, 0]
        sol = Solution()
        res = sol.nextPermutation(arr)
        self.assertTrue(res)
        self.assertEqual(arr, [1, 0, 0])

    def test_leftover_bidders(self):
        from datastructure.story_teller import MySpecialQueue
        res = MySpecialQueue.leftover_bidders([1, 2, 3, 4, 5, 6, 7, 8, 9], 2)
        self.assertEqual(res, [1, 2, 3, 4, 5, 6, 7])

    def test_get_moves(self):
        from datastructure.story_teller import get_moves, Combo
        res = get_moves(target=Combo(4, 7, 6), deadends={Combo(6, 6, 6)})
        self.assertEqual(res, 14)

    @patch('sys.stdout', new_callable=StringIO)
    def test_greedy_activities(self, mock_stdout):
        from datastructure.story_teller import greedy
        activities = [
            ["A1", 0, 6],
            ["A2", 3, 4],
            ["A3", 1, 2],
            ["A4", 5, 8],
            ["A5", 5, 7],
            ["A6", 8, 9]
        ]
        greedy.printMaxActivities(activities)
        output = mock_stdout.getvalue().strip().split('\n')
        self.assertEqual(output, ["A3", "A2", "A5", "A6"])

    def test_greedy_coin_change(self):
        from datastructure.story_teller import greedy
        # Runs without error
        greedy.coinChange(201, [1, 2, 5, 20, 50, 100])

    def test_solve(self):
        from datastructure.story_teller import solve, BASE_DIR
        import os
        tree = [[1, 2, 3], [1, 4, 2], [2, 5, 6], [3, 4, 1]]
        queries = [[1, 1], [1, 2], [2, 3], [2, 5], [1, 6]]
        solve(tree, queries)
        filepath = os.path.join(BASE_DIR, "output_graph_supersum.txt")
        self.assertTrue(os.path.exists(filepath))
        with open(filepath, "r") as f:
            content = f.read().strip().split('\n')
        self.assertEqual(content, ["1", "3", "5", "5", "10"])

    def test_kundu_and_tree(self):
        # N=5 nodes, black edges: 1-2, 4-5. Red edges: 2-3, 3-4.
        edges = [
            (1, 2, 'b'),
            (2, 3, 'r'),
            (3, 4, 'r'),
            (4, 5, 'b')
        ]
        ans = KunduAndTree.solve(5, edges)
        self.assertEqual(ans, 4)



    def test_board_solve(self):
        from datastructure.story_teller import Board
        count, path = Board.solve('124356870')
        self.assertEqual(count, 22)
        self.assertEqual(path, 'DRRDLUULDDRRULLURRDLLU')

    def test_picking_numbers(self):
        from datastructure.story_teller import pickingNumbers
        self.assertEqual(pickingNumbers([]), 3)

    def test_hourglass_sum(self):
        from datastructure.story_teller import hourglassSum
        arr = [
            [1, 2, 3, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 2, 4, 4, 0],
            [0, 0, 0, 2, 0, 0],
            [0, 0, 1, 2, 4, 0]
        ]
        self.assertEqual(hourglassSum(arr), 19)

    @patch('sys.stdout', new_callable=StringIO)
    def test_graph_distances(self, mock_stdout):
        from datastructure.story_teller import Graph
        g = Graph(4)
        g.connect(0, 1)
        g.connect(0, 2)
        g.find_all_distances(0)
        output = mock_stdout.getvalue().strip()
        self.assertEqual(output, "6 6 -1")

    def test_playing_deck(self):
        from datastructure.story_teller import PlayingDeck
        deck = PlayingDeck()
        self.assertEqual(len(deck), 52)
        self.assertTrue(hasattr(deck[0], 'suit'))
        self.assertTrue(hasattr(deck[0], 'rank'))

    def test_problem308(self):
        from datastructure.story_teller import problem308
        self.assertEqual(problem308.solve(['F', '|', 'T', '&', 'T']), 2)

    @patch('sys.stdin', StringIO("7\n3 4 1 2 1 5 1\n"))
    @patch('sys.stdout', new_callable=StringIO)
    def test_stock_gain(self, mock_stdout):
        from datastructure.story_teller import Stock
        Stock.stock_gain()
        output = mock_stdout.getvalue().strip()
        self.assertIn("gain 4", output)

    def test_prime_sieve_showcase(self):
        from datastructure.story_teller import PrimeSieveShowcase
        self.assertTrue(PrimeSieveShowcase.is_prime(991))
        self.assertFalse(PrimeSieveShowcase.is_prime(4))
        
        primes_900_1000 = PrimeSieveShowcase.prime_interval(900, 1000)
        self.assertIn(991, primes_900_1000)
        
        primes_10_100 = PrimeSieveShowcase.primesInRange(10, 100)
        self.assertIn(11, primes_10_100)
        self.assertIn(97, primes_10_100)

    def test_nelder_mead_showcase(self):
        from datastructure.story_teller import NelderMeadShowcase
        res = NelderMeadShowcase.optimize()
        self.assertIsNotNone(res)
        self.assertTrue(hasattr(res, 'x'))

    def test_power_sum_ntt(self):
        from datastructure.story_teller import PowerSumNTT
        ans = PowerSumNTT.get_answer([0, 1, 2, 1], [0, 2, 3, 4], 3, 2, 2)
        self.assertEqual(ans, 65)

    def test_solution_find_judge(self):
        from datastructure.story_teller import Solution
        sol = Solution()
        self.assertEqual(sol.findJudge(2, [[1, 2]]), 2)
        self.assertEqual(sol.findJudge(3, [[1, 3], [2, 3]]), 3)
        self.assertEqual(sol.findJudge(3, [[1, 3], [2, 3], [3, 1]]), -1)

    @patch('sys.stdout', new_callable=StringIO)
    def test_array_right_rotate(self, mock_stdout):
        from datastructure.story_teller import ArrayProblems
        ArrayProblems.RightRotate([1, 2, 3, 4, 5], 5, 2)
        output = mock_stdout.getvalue().strip()
        self.assertEqual(output, "4 5 1 2 3")

    def test_fib_dn(self):
        from datastructure.story_teller import FibonacciShowcase
        self.assertEqual(FibonacciShowcase.fib_dn(10, {0: 1, 1: 1}), 89)


class TestNumberNice(unittest.TestCase):

    def test_all_free_n6(self):
        self.assertEqual(story_teller.number_nice([-1] * 6), 60)

    def test_all_zeros_n4(self):
        self.assertEqual(story_teller.number_nice([0, 0, 0, 0]), 1)

    def test_bad_first(self):
        self.assertEqual(story_teller.number_nice([1, -1, -1]), 0)


class TestArrayProblems(unittest.TestCase):

    def test_mode_sorted(self):
        self.assertEqual(ArrayProblems.mode_sorted([1, 2, 3, 3, 3, 4, 4, 5, 5, 6]), 3)

    def test_mode_unsorted(self):
        self.assertEqual(ArrayProblems.mode_unsorted([3, 1, 3, 2, 3]), 3)


class TestStock(unittest.TestCase):

    def test_k1(self):
        self.assertEqual(Stock.max_profit_k_transactions([2, 4, 1, 7], 1), 6)

    def test_k2(self):
        self.assertEqual(Stock.max_profit_k_transactions([2, 4, 1, 7], 2), 8)

    def test_flat(self):
        self.assertEqual(Stock.max_profit_k_transactions([5, 5, 5, 5], 2), 0)

    def test_vic(self):
        self.assertEqual(Stock.max_profit_k_transactions([225, 224, 407, 221, 259, 403], 3), 365)


class TestFibonacciCorrectness(unittest.TestCase):

    def setUp(self):
        self.expected_fibs = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]

    def test_fibo(self):
        got = [FibonacciShowcase.fibo(n) for n in range(1, 16)]
        self.assertEqual(got, self.expected_fibs)

    def test_simple_fibo(self):
        got = [FibonacciShowcase.simple_fibo(n) for n in range(1, 16)]
        self.assertEqual(got, self.expected_fibs)

    def test_dyna_fibo(self):
        got = [FibonacciShowcase.dyna_fibo(n, {1: 1, 2: 1}) for n in range(1, 16)]
        self.assertEqual(got, self.expected_fibs)

    def test_dyna_fibo2(self):
        got = [FibonacciShowcase.dyna_fibo2(n) for n in range(1, 16)]
        self.assertEqual(got, self.expected_fibs)

    def test_fibonacci_bu(self):
        got = [FibonacciShowcase.fibonacci_bu(n) for n in range(1, 16)]
        self.assertEqual(got, self.expected_fibs)

    def test_bin_fibo(self):
        got = [FibonacciShowcase.bin_fibo(n) for n in range(1, 16)]
        self.assertEqual(got, self.expected_fibs)

    def test_my_fib(self):
        got = [FibonacciShowcase.my_fib(n, {}) for n in range(1, 16)]
        self.assertEqual(got, self.expected_fibs)


class TestFibonacciBenchmark(unittest.TestCase):

    def test_benchmark(self):
        # Simply run to verify no exceptions
        FibonacciShowcase.compare(N=200, count=20)


class TestDynamicProgramming(unittest.TestCase):

    def test_divide_min_diff(self):
        s1, s2 = DynamicProgramming.divide_numbers([5, 10, 15, 20, 25])
        self.assertEqual(abs(sum(s1) - sum(s2)), 5)

    def test_lis(self):
        self.assertEqual(DynamicProgramming.lis([1, 2, 5, 4, 6, 2]), 4)

    def test_lis_two(self):
        self.assertEqual(DynamicProgramming.lis([10, 9, 2, 5, 3, 7, 101, 18]), 4)

    def test_lis_recursive(self):
        self.assertEqual(DynamicProgramming.lis_recursive([1, 2, 5, 4, 6, 2]), 4)


class TestSearchAndCount(unittest.TestCase):

    def test_binary_search_found(self):
        self.assertTrue(SearchAndCount.binary_search([1, 3, 5, 7, 9], 5))

    def test_binary_search_not_found(self):
        self.assertFalse(SearchAndCount.binary_search([1, 3, 5, 7, 9], 4))

    def test_count_pairs(self):
        self.assertEqual(SearchAndCount.count_pairs([1, 3, 5], [2, 4, 6], 7), 3)

    def test_count_pairs3(self):
        self.assertEqual(SearchAndCount.count_pairs3([1, 2], [3, 4], [4, 5, 6]), 4)


class TestGeometryProblems(unittest.TestCase):

    def test_rect_intersection(self):
        r1 = {"top_left": (1, 4), "dimensions": (3, 3)}
        r2 = {"top_left": (0, 5), "dimensions": (4, 3)}
        self.assertEqual(GeometryProblems.rectangle_intersection(r1, r2), 6)

    def test_no_intersection(self):
        r3 = {"top_left": (0, 0), "dimensions": (1, 1)}
        r4 = {"top_left": (5, 5), "dimensions": (1, 1)}
        self.assertEqual(GeometryProblems.rectangle_intersection(r3, r4), 0)


class TestClassicPuzzles(unittest.TestCase):

    def test_find_ith_digit_1(self):
        self.assertEqual(ClassicPuzzles.find_ith_digit(1), 1)

    def test_find_ith_digit_6(self):
        self.assertEqual(ClassicPuzzles.find_ith_digit(6), 3)

    def test_find_ith_digit_10(self):
        self.assertEqual(ClassicPuzzles.find_ith_digit(10), 4)

    def test_last_unique_found(self):
        self.assertEqual(ClassicPuzzles.last_unique_character("slideeducation"), "n")

    def test_last_unique_none(self):
        self.assertEqual(ClassicPuzzles.last_unique_character("aabb"), "none")

    def test_pyramid_max_sum(self):
        pyr = [[3], [7, 4], [2, 4, 6], [8, 5, 9, 3]]
        self.assertEqual(ClassicPuzzles.highest_sum_pyramid(pyr), 23)

    def test_matrix_range_count(self):
        mat = [
            [1,  3,  7, 10, 15, 20],
            [2,  6,  9, 14, 22, 25],
            [3,  8, 10, 15, 25, 30],
            [10, 11, 12, 23, 30, 35],
            [20, 25, 30, 35, 40, 45],
        ]
        self.assertEqual(ClassicPuzzles.count_matrix_range(mat, 1, 1, 3, 3), 14)


class TestPythonPatterns(unittest.TestCase):

    def test_regex_examples(self):
        PythonPatterns.regex_examples()

    def test_chain_examples(self):
        PythonPatterns.chain_examples()


if __name__ == '__main__':
    unittest.main()


class TestSnippetExamples(unittest.TestCase):

    def test_left_rotate(self):
        self.assertEqual(SnippetExamples.left_rotate([1,2,3,4,5], 4), [5,1,2,3,4])
        self.assertEqual(SnippetExamples.left_rotate([1,2,3,4,5], 0), [1,2,3,4,5])

    def test_invert_dict(self):
        result = SnippetExamples.invert_dict({"a":"x","b":"x","c":"y"})
        self.assertEqual(sorted(result["x"]), ["a","b"])
        self.assertEqual(result["y"], ["c"])

    def test_reduce_product(self):
        self.assertEqual(SnippetExamples.reduce_product([1,2,3,4]), 24)

    def test_diagonal_difference(self):
        self.assertEqual(SnippetExamples.diagonal_difference([[11,2,4],[4,5,6],[10,8,-12]]), 15)

    def test_palindrome(self):
        self.assertTrue(SnippetExamples.palindrome("abcba"))
        self.assertFalse(SnippetExamples.palindrome("hello"))

    def test_array_manipulation(self):
        self.assertEqual(SnippetExamples.array_manipulation(5, [[1,2,100],[2,5,100],[3,4,100]]), 200)


class TestSubsetProblems(unittest.TestCase):

    def test_isSubsetSum_recursive(self):
        self.assertTrue(SubsetProblems.isSubsetSum_recursive([3,34,4,12,5,2], 6, 9))
        self.assertFalse(SubsetProblems.isSubsetSum_recursive([3,34,4,12,5,2], 6, 30))

    def test_isSubsetSum_dp(self):
        self.assertTrue(SubsetProblems.isSubsetSum_dp([3,34,4,12,5,2], 6, 9))
        self.assertFalse(SubsetProblems.isSubsetSum_dp([3,34,4,12,5,2], 6, 30))

    def test_subset(self):
        result = SubsetProblems.subset([3,4,2,1], 5)
        self.assertIn((4,1), result)
        self.assertIn((3,2), result)

    def test_k_partition(self):
        result = SubsetProblems.k_partition([7,3,5,12,2,1,5,3,8,4,6,4], 5)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 5)
        part_sum = sum(result[0])
        for part in result:
            self.assertEqual(sum(part), part_sum)

    def test_k_partition_impossible(self):
        self.assertIsNone(SubsetProblems.k_partition([1,2,3], 4))


class TestDifferenceArray(unittest.TestCase):

    def test_initialize_and_reconstruct(self):
        A = [10, 5, 20, 40]
        D = DifferenceArray.initialize(A)
        DifferenceArray.update(D, 0, 1, 10)
        result = DifferenceArray.reconstruct(A, D)
        self.assertEqual(result, [20, 15, 20, 40])

    def test_multiple_updates(self):
        A = [10, 5, 20, 40]
        D = DifferenceArray.initialize(A)
        DifferenceArray.update(D, 0, 1, 10)
        DifferenceArray.update(D, 1, 3, 20)
        DifferenceArray.update(D, 2, 2, 30)
        result = DifferenceArray.reconstruct(A, D)
        self.assertEqual(result, [20, 35, 70, 60])


class TestStockProfit(unittest.TestCase):
    prices = [50, 25, 12, 4, 3, 10, 1, 100]
    k = 2

    def test_max_profit_kn(self):
        self.assertEqual(StockProfit.max_profit_kn(self.prices, self.k), 106)

    def test_max_profit_kn_optimized(self):
        self.assertEqual(StockProfit.max_profit_kn_optimized(self.prices, self.k), 106)

    def test_empty(self):
        self.assertEqual(StockProfit.max_profit_kn([], 2), 0)
        self.assertEqual(StockProfit.max_profit_kn_optimized([], 2), 0)


class TestRiverSizes(unittest.TestCase):

    def test_river_sizes(self):
        matrix = [
            [1, 0, 0, 1, 0],
            [1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 1, 0],
        ]
        self.assertEqual(sorted(RiverSizes.river_sizes(matrix)), sorted([1, 2, 2, 2, 5]))

    def test_no_rivers(self):
        self.assertEqual(RiverSizes.river_sizes([[0,0],[0,0]]), [])


class TestArrayUtils(unittest.TestCase):

    def test_largest_range(self):
        self.assertEqual(ArrayUtils.largest_range([1,11,3,0,15,5,2,4,10,7,12,6]), [0, 7])

    def test_birthday(self):
        self.assertEqual(ArrayUtils.birthday([1,2,1,3,2], 3, 2), 2)

    def test_find_root_bisection(self):
        result = ArrayUtils.find_root_bisection(lambda x: x**2 - x - 1, 1, 2, 25)
        self.assertAlmostEqual(result, (1 + 5**0.5)/2, places=5)

    def test_h_index(self):
        self.assertEqual(ArrayUtils.h_index([4,1,0,2,3]), 2)
        self.assertEqual(ArrayUtils.h_index([0,0,0]), 0)

    def test_sliding_window_median(self):
        result = ArrayUtils.sliding_window_median([-1,5,13,8,2,3,3,1], 3)
        self.assertEqual(result[0], 5.0)

    def test_island_perimeter(self):
        board = [[0,1,1,0],[1,1,1,0],[0,1,1,0],[0,0,1,0]]
        self.assertEqual(ArrayUtils.island_perimeter(board), 14)

    def test_group_anagrams(self):
        result = ArrayUtils.group_anagrams(["eat","ate","apt","pat","tea","now"])
        groups = [sorted(g) for g in result]
        self.assertIn(sorted(["eat","ate","tea"]), groups)
        self.assertIn(sorted(["apt","pat"]), groups)
        self.assertIn(["now"], groups)

    def test_compress_array_continuous(self):
        self.assertEqual(ArrayUtils.compress_array_continuous([1,1,1,4,4,3]), [1,3,4,2,3,1])

    def test_compress_array_any(self):
        self.assertEqual(ArrayUtils.compress_array_any([1,1,1,4,4,3,1,3]), [1,4,4,2,3,2])

    def test_longest_contiguous_history(self):
        u1 = ["/home","/register","/login","/user","/one","/two"]
        u2 = ["/home","/red","/login","/user","/one","/pink"]
        self.assertEqual(ArrayUtils.longest_contiguous_history(u1, u2), ["/login","/user","/one"])


class TestGraphProblems(unittest.TestCase):

    def test_components_in_graph(self):
        gb = [[1,6],[2,7],[3,8],[4,9],[2,6]]
        mn, mx = GraphProblems.components_in_graph(gb)
        self.assertLessEqual(mn, mx)

    def test_minimum_average(self):
        self.assertEqual(GraphProblems.minimum_average([[0,3],[1,9],[2,5]]), 8)


class TestUnionFind(unittest.TestCase):

    def test_make_set_and_find(self):
        ds = DisjointSet()
        ds.make_set(1)
        self.assertEqual(ds.find(1), 1)

    def test_union(self):
        ds = DisjointSet()
        ds.make_set(1)
        ds.make_set(2)
        ds.union(1, 2)
        self.assertEqual(ds.find(1), ds.find(2))

    def test_size_after_union(self):
        ds = DisjointSet()
        for x in [1, 2, 3]:
            ds.make_set(x)
        ds.union(1, 2)
        ds.union(2, 3)
        self.assertEqual(ds.get_size(1), 3)


class TestBSTShowcase(unittest.TestCase):

    def test_is_bst_valid(self):
        BSTNode = BSTShowcase.BSTNode
        root = BSTNode(10)
        root.left_child = BSTNode(6)
        root.right_child = BSTNode(15)
        root.left_child.left_child = BSTNode(4)
        root.left_child.right_child = BSTNode(9)
        self.assertTrue(BSTShowcase.is_bst(root))

    def test_is_bst_invalid(self):
        BSTNode = BSTShowcase.BSTNode
        root = BSTNode(10)
        root.left_child = BSTNode(20)  # violates BST
        self.assertFalse(BSTShowcase.is_bst(root))

    def test_size(self):
        BSTNode = BSTShowcase.BSTNode
        root = BSTNode(10)
        root.left_child = BSTNode(6)
        root.right_child = BSTNode(15)
        self.assertEqual(BSTShowcase.size(root), 3)

    def test_largest_bst_subtree_naive_pure_bst(self):
        BSTNode = BSTShowcase.BSTNode
        root = BSTNode(10)
        root.left_child = BSTNode(6)
        root.right_child = BSTNode(15)
        root.left_child.left_child = BSTNode(4)
        root.left_child.right_child = BSTNode(9)
        root.right_child.left_child = BSTNode(12)
        root.right_child.right_child = BSTNode(24)
        result = BSTShowcase.largest_bst_subtree_naive(root)
        self.assertIsNotNone(result)
        self.assertEqual(result.data, 10)

    def test_largest_bst_subtree_optimized_pure_bst(self):
        BSTNode = BSTShowcase.BSTNode
        root = BSTNode(10)
        root.left_child = BSTNode(6)
        root.right_child = BSTNode(15)
        root.left_child.left_child = BSTNode(4)
        root.left_child.right_child = BSTNode(9)
        root.right_child.left_child = BSTNode(12)
        root.right_child.right_child = BSTNode(24)
        result = BSTShowcase.largest_bst_subtree_optimized(root)
        self.assertIsNotNone(result)
        self.assertEqual(result.data, 10)


class TestCameraCoverSolution(unittest.TestCase):

    def test_single_chain(self):
        root = CameraCoverSolution(0)
        root.left = CameraCoverSolution(0)
        root.left.left = CameraCoverSolution(0)
        self.assertEqual(root.min_camera_cover(), 1)

    def test_no_children(self):
        root = CameraCoverSolution(0)
        self.assertEqual(root.min_camera_cover(), 1)


class TestSynonymQueries(unittest.TestCase):
    synonyms = [("big","large"),("eat","consume")]
    queries = [("He wants to eat big food.","He wants to consume large food.")]

    def test_naive(self):
        result = SynonymQueries.solve_naive(self.synonyms, self.queries)
        self.assertEqual(result, [True])

    def test_naive_not_equivalent(self):
        result = SynonymQueries.solve_naive(
            [("big","large")], [("He is big.","He is huge.")]
        )
        self.assertEqual(result, [False])

    def test_disjoint_set_transitive(self):
        # big->large->huge is transitive; words must have no punctuation for word-level matching
        result = SynonymQueries.solve_disjoint_set(
            [("big","large"),("large","huge")],
            [("He is big","He is huge")]
        )
        self.assertEqual(result, [True])

    def test_disjoint_set_naive_no_transitive(self):
        # naive should return False for same transitive case
        result = SynonymQueries.solve_naive(
            [("big","large"),("large","huge")],
            [("He is big.","He is huge.")]
        )
        self.assertEqual(result, [False])

    def test_different_lengths(self):
        result = SynonymQueries.solve_disjoint_set(
            [], [("one two","one")]
        )
        self.assertEqual(result, [False])


class TestBSTMatrixVector(unittest.TestCase):
    def test_bst_vector_operations(self):
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Data_Structures_Algorithms_In_Python-master", "Tree", "BinarySearchTree"))
        from BST_MatrixVector import BSTVector
        
        vec = BSTVector(10)
        vec.set(2, 5.0)
        vec.set(7, 3.0)
        
        # Verify norm2 calculation
        self.assertEqual(vec.norm2, 34.0)
        
        # Verify elements access
        self.assertEqual(vec.get(2), 5.0)
        self.assertEqual(vec.get(7), 3.0)
        self.assertEqual(vec.get(1), 0.0)
        
        # Verify bounds check
        with self.assertRaises(IndexError):
            vec.get(10)
            
        # Verify sampling
        sampled = vec.sample(seed=42)
        self.assertIn(sampled, [2, 7])

    def test_bst_matrix_operations(self):
        import sys
        import os
        import numpy as np
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Data_Structures_Algorithms_In_Python-master", "Tree", "BinarySearchTree"))
        from BST_MatrixVector import BSTVector, BSTMatrix
        
        # Test vectors norms validation
        v = BSTVector(100)
        a = np.zeros((100,))
        import random
        random.seed(42)
        for _ in range(100):
            index = random.randint(0, 99)
            value = random.random()
            v.set(index, value)
            a[index] = value
        self.assertLess(abs(v.norm2 - np.sum(np.square(a))), 1e-9)

        # Test matrix norms validation
        m = BSTMatrix(50, 20)
        b = np.zeros((50, 20))
        for _ in range(500):
            row = random.randint(0, 49)
            col = random.randint(0, 19)
            value = random.random()
            m.set(row, col, value)
            b[row, col] = value
            
        self.assertLess(abs(m.frob_norm2 - np.sum(np.square(b))), 1e-9)
        self.assertLess(abs(m.get_row_norm(5) - np.linalg.norm(b[5])), 1e-9)
        
        # Verify row sampling
        row_sampled = m.sample_row_norms()
        self.assertTrue(0 <= row_sampled < 50)
        
        col_sampled = m.sample_row(row_sampled)
        self.assertTrue(0 <= col_sampled < 20)


