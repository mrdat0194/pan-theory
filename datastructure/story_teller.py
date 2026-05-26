#!/bin/python3
import os  # noqa: E402
from math import factorial  # noqa: E402
import itertools  # noqa: E402
from collections import defaultdict, Counter, namedtuple  # noqa: E402
from functools import cmp_to_key, reduce  # noqa: E402
import heapq  # noqa: E402
from typing import Set, List  # noqa: E402
import re  # noqa: E402
import operator  # noqa: E402
import random  # noqa: E402
import queue  # noqa: E402
from queue import Queue  # noqa: E402

from my_functions.timer import print_param, timer  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class story_teller:
    @timer
    @print_param("output_miniswap.txt", BASE_DIR)
    def minimumSwaps(arr):
        """
        Note: Minimum swap cannot solve wholy as minimumBribe
        swap to get a increasing order
        Given array
        After swapping  we get 1 3 4 2
        After swapping  we get 1 4 3 2
        After swapping  we get 1 2 3 4
        So, we need a minimum of 3 swaps to sort the array in ascending order

        n = int(input())

        arr = list(map(int, input().rstrip().split()))

        res = story_teller.minimumSwaps(arr)

        Example:
        # 4
        # 4 3 1 2
        -> 1 2 3 4

        :param arr: array = list [1,2,3] { 1: "a", 2..} ( )
        :return:
        """
        ref_arr = sorted(arr)

        index_dict = {v: i for i,v in enumerate(arr)}

        swaps = 0

        for i,v in enumerate(arr):
            correct_value = ref_arr[i]
            if v != correct_value:
                to_swap_ix = index_dict[correct_value]
                arr[to_swap_ix],arr[i] = arr[i], arr[to_swap_ix]
                index_dict[v] = to_swap_ix
                index_dict[correct_value] = i
                swaps += 1
        return swaps

    def minimumBribes(Q):
        '''
        list of people in the queue, sequentially bribe the one in front of them
            [1,2,3,4,5] -> [4,1,2,3,5] -> [5,4,1,2,3] 7 steps -> [5,4,3,1,2] 9 steps
            Thought:

                Easy:
                [5,1,2,3,4]
                -> init: [4,0,1,2,3] -> enum: [0,1,2,3,4]

                Hard:
                [5,4,3,1,2]
                init:
                [4,3,2,0,1]
                enum:
                [0,1,2,3,4]

            at least how many bribes happened.

            t = int(input())

            for t_itr in range(t):
                n = int(input())

                q = list(map(int, input().rstrip().split()))

            story_teller.minimumBribes(q)
        2
        5
        2 1 5 3 4
        5
        2 5 1 3 4

        :param Q: arr
        :return:
        '''
        #
        # initialize the number of moves
        moves = 0
        #
        # decrease Q by 1 to make index-matching more intuitive
        # so that our values go from 0 to N-1, just like our
        # indices.  (Not necessary but makes it easier to
        # understand.)
        # Init:
        Q = [P-1 for P in Q]
        #
        # Loop through each person (P) in the queue (Q)
        for i,P in enumerate(Q):
            # i is the current position of P, while P is the
            # original position of P.
            #
            # First check if any P is more than two ahead of
            # its original position
            # if P - i > 5:
            #     print("Too chaotic")
            #     return
            #
            # From here on out, we don't care if P has moved
            # forwards, it is better to count how many times
            # P has RECEIVED a bribe, by looking at who is
            # ahead of P.  P's original position is the value
            # of P.
            # Anyone who bribed P cannot get to higher than
            # one position in front if P's original position,
            # so we need to look from one position in front
            # of P's original position to one in front of P's
            # current position, and see how many of those
            # positions in Q contain a number large than P.
            # In other words we will look from P-1 to i-1,
            # which in Python is range(P-1,i-1+1), or simply
            # range(P-1,i).  To make sure we don't try an
            # index less than zero, replace P-1 with
            # max P-1,0)
            # change 100 accordingly
            for j in range(max(P-100,0),i):

                if Q[j] > P:
                    moves += 1
        return moves

    @print_param("output_twoString.txt", BASE_DIR)
    def twoStrings(s1, s2):
        '''
            String that intersect
            # 1
            # hello
            # world
        :param s1:
        :param s2:
        :return:
        '''
        s1 = set(s1)
        s2 = set(s2)
        result = s1.intersection(s2)
        if not result:
            print('NO')
            return 'NO'
        else:
            print('YES')
            return 'YES'

    @print_param("output_sherlockAndAnagrams.txt", BASE_DIR)
    def sherlockAndAnagrams(string):
        '''
        # 2
        # ifailuhkqq
        # kkkk
        For the first query, we have anagram pairs  and  at positions  and  respectively.

        For the second query:
        There are 6 anagrams of the form  at positions  and .
        There are 3 anagrams of the form  at positions  and .
        There is 1 anagram of the form  at position .
        :param string:
        :return:
        '''
        buckets = {}
        for i in range(len(string)):
            for j in range(1, len(string) - i + 1):
                key = frozenset(Counter(string[i:i+j]).items()) # O(N) time key extract
                buckets[key] = buckets.get(key, 0) + 1
        count = 0
        # print(buckets)
        for key in buckets:
            count += buckets[key] * (buckets[key]-1) // 2
        return count

    @print_param("output_makeAna.txt", BASE_DIR)
    def makeAnagram(a, b):
        """
             How many deletion to make two string anagram

             a = input()
             b = input()
             story_teller.makeAnagram(a,b)

        :param a,b:
             cde
             abc
        :return:
        """

        ct_a = Counter(a)
        ct_b = Counter(b)
        ct_a.subtract(ct_b)
        return sum(abs(i) for i in ct_a.values())

    def checkMagazine(magazine, note):
        '''
        6 4
        give me one grand today night
        give one grand today
        completely contain in magazine as in note

            mn = input().split()

            m = int(mn[0])

            n = int(mn[1])

            magazine = input().rstrip().split()

            note = input().rstrip().split()

            print(ransom_note(magazine, note))
            print(checkMagazine(magazine, note))

        :param magazine:
        :param note:
        :return:
        '''
        d = {}
        for word in magazine:
            d.setdefault(word, 0)
            d[word] += 1
        for word in note:
            if word in d:
                d[word] -= 1
            else:
                return False

        return all([x >= 0 for x in d.values()])


    def ransom_note(magazine, ransom):
        return (Counter(ransom) - Counter(magazine)) == {}

    @timer
    @print_param("output_sock.txt", BASE_DIR)
    def sockMerchant(n, ar):
        """
            sum of pair of sock
            n = int(input())
            ar = list(map(int, input().rstrip().split()))

        :param n:how many socks : 9
        :param ar: 10 20 20 10 10 30 50 10 20
        :return: sum of pair of sock
        """
        sum=0
        for values in Counter(ar).values():
            sum+=values//2
        return sum



    class newNode:
        def __init__(self, data):
            self.data = data
            self.left = None
            self.right = None

    def largestBST(node):
        # Set the initial values for calling largestBSTUtil()
        Min = [float('inf')]
        Max = [float('-inf')]
        max_size = [0]
        is_bst = [0]
        story_teller.largestBSTUtil(node, Min, Max, max_size, is_bst)
        return max_size[0]

    def largestBSTUtil(node, min_ref, max_ref, max_size_ref, is_bst_ref):
        if node == None:
            is_bst_ref[0] = 1 # An empty tree is BST
            return 0 # Size of the BST is 0

        Min = float('inf')
        left_flag = False
        right_flag = False
        ls, rs = 0, 0

        max_ref[0] = float('-inf')
        ls = story_teller.largestBSTUtil(node.left, min_ref, max_ref, max_size_ref, is_bst_ref)
        if is_bst_ref[0] == 1 and node.data > max_ref[0]:
            left_flag = True

        Min = min_ref[0]
        min_ref[0] = float('inf')
        rs = story_teller.largestBSTUtil(node.right, min_ref, max_ref, max_size_ref, is_bst_ref)
        if is_bst_ref[0] == 1 and node.data < min_ref[0]:
            right_flag = True

        if Min < min_ref[0]:
            min_ref[0] = Min
        if node.data < min_ref[0]:
            min_ref[0] = node.data
        if node.data > max_ref[0]:
            max_ref[0] = node.data

        if left_flag and right_flag:
            if ls + rs + 1 > max_size_ref[0]:
                max_size_ref[0] = ls + rs + 1
            return ls + rs + 1
        else:
            is_bst_ref[0] = 0
            return 0


    def twoSumHashing(num_arr, pair_sum):
        hashTable = {}
        for i in range(len(num_arr)):
            complement = pair_sum - num_arr[i]
            if complement in hashTable:
                print("Pair with sum", pair_sum,"is: (", num_arr[i],",",complement,")")
            hashTable[num_arr[i]] = num_arr[i]

    @print_param("output_freqQuery.txt", BASE_DIR)
    def freqQuery(queries):

        """
            command:
            1 - x : Insert x in your data structure.
            2 - y : Delete one occurence of y from your data structure, if present.
            3 - z : Check if any integer is present whose frequency is exactly . If yes, print 1 else 0

            q = int(input().strip())
            queries = []
            for _ in range(q):
                queries.append(list(map(int, input().rstrip().split())))

            story_teller.freqQuery(queries)

        :param queries:
        8
        1 5
        1 6
        3 2
        1 10
        1 10
        1 6
        2 5
        3 2
        :return:[]
        """

        results = []
        lookup = dict()
        freqs = defaultdict(set)
        for command, value in queries:
            freq = lookup.get(value, 0)
            if command == 1:
                lookup[value] = freq + 1
                freqs[freq].discard(value)
                freqs[freq + 1].add(value)
            elif command == 2:
                lookup[value] = max(0, freq - 1)
                freqs[freq].discard(value)
                freqs[freq - 1].add(value)
            elif command == 3:
                results.append(1 if freqs[value] else 0)
        print(freqs)
        return results

    @print_param("output_valleycount.txt", BASE_DIR)
    def countingValleys(n, s):
        """
            countingValleys
            n = int(input())
            s = input()
            story_teller.countingValleys(n, s)

        :param :n 8
        :param s: UDDDUDUU
        :return: up and down how many valley
        """
        UD = {'U': 1, 'D': -1}
        sea_level = 0
        valley = 0
        for step in s:
            sea_level = sea_level + UD[step]
            if not sea_level and step == 'U':
                valley += 1
        return valley

    def construct_pyramid(lenMax):
        if lenMax % 2 == 1:
            peak = lenMax//2 + 1
            x_left = [x for x in range(1,peak)]
            x_right = list(reversed(x_left))
            pyramid = x_left + [peak] + x_right
            return pyramid

    def pyramid_build(n, ar):
        """
         You have N stones in a row, and would like to create from them a pyramid.
         This pyramid should be constructed such that the height of each stone increases by one until reaching the tallest stone,
         after which the heights decrease by one. In addition, the start and end stones of the pyramid should each be one stone high.
         You can change the height of any stone by paying a cost of 1 unit to lower its height by 1, as many times as necessary.
         Given this information, determine the lowest cost method to produce this pyramid.
         For example, given the stones [1, 1, 3, 3, 2, 1], the optimal solution is to pay 2 to create [0, 1, 2, 3, 2, 1]
            n = int(input())
            ar = list(map(int, input().rstrip().split()))
            story_teller.pyramid_build(n, ar)
        :param n: 6
        :param ar: 1 1 3 3 2 1
        1 1 1 5 1
        :return: [0, 1, 2, 3, 2, 1]

        """
        lenStone = n
        stones = ar
        lenMax = lenStone if lenStone % 2 else lenStone - 1
        cost = 0
        while lenMax > 0:
            pyramid = story_teller.construct_pyramid(lenMax)

            for offset in (0, lenStone - lenMax):
                valid = True
                for enum_index, enum_val in enumerate(pyramid):
                    stone_index = enum_index + offset
                    if enum_val > stones[stone_index]:
                        valid = False
                        break

                if valid:
                    result = [0]*offset + pyramid +[0]*(lenStone-offset-lenMax)
                    cost = sum([x[0] - x[1] for x in zip(stones,result)])
                    print(cost)
                    return result

            lenMax -= 2
        return []


    @print_param("output_repeatedstrings.txt", BASE_DIR)
    def repeatedString(s, n):
        """
            repeatedstrings
            s = input()
            n = int(input())
            result = story_teller.repeatedString(s, n)

        :param s: aba
        :param n: 10 of letters contain aba aba aba / a (phần dư)
        :return: how many a
        """
        return s.count("a") * (n // len(s)) + s[:n % len(s)].count("a")

    def countTriplets(arr, r):
        """
            Counting how many triplet of the group of exponential
            5 5
            1 5 5 25 125
            n,r = map(int,input().split())
            arr = list(map(int,input().split()))
            print(countTriplets(arr, r))

        :param arr: increasing seq
        :param r: num of exp
        :return: Counttriplet how many triple that increase


        """
        a = Counter(arr)
        b = Counter()
        s = 0
        for i in arr:
            # số chia
            j = i//r
            # số nhân
            k = i*r
            a[i]-=1
            # số lưu tạo đc bao nhiêu cặp 3
            if b[j] and a[k] and not i%r:
                s+=b[j]*a[k]
            b[i]+=1
        return s

    def formingMagicSquare(s):
        """
        convert it into a magic square at minimal cost. Print this cost on a new line.

        Note: The resulting magic square must contain distinct integers in the inclusive range [1-9]
            s = []
            for s_i in range(3):
                s += [int(s_temp) for s_temp in input().strip().split(' ')]
            result = formingMagicSquare(s)

        4 9 2
        3 5 7
        8 1 5
        :param s:
        :return:
        """
        squares = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 4, 8], [2, 4, 6],
                   [0, 3, 6], [1, 4, 7], [2, 5, 8]]
        def ismagic(xs):
            for sq in squares:
                if sum(xs[s] for s in sq) != 15:
                    return False
            return True

        gen = filter(ismagic, list(itertools.permutations(range(1,10))))

        return min(sum(abs(x - y) for x, y in zip(gl, s)) for gl in gen)


    # ── from Biggest_BST2.py ──────────────────────────────────────────────────
    def largestBSTBT(root):
        """
        Alternative approach to find largest BST in a Binary Tree.
        Returns tuple (size, max_val, min_val, bst_size, is_bst).

        Example:
            root = story_teller.newNode(60)
            root.left = story_teller.newNode(65)
            root.right = story_teller.newNode(70)
            root.left.left = story_teller.newNode(50)
            print(story_teller.largestBSTBT(root)[3])

        :param root: binary tree root (story_teller.newNode)
        :return: list [size, max, min, bst_size, is_bst]
        """
        INT_MIN = -2147483648
        INT_MAX = 2147483647
        if root is None:
            return [0, INT_MIN, INT_MAX, 0, True]
        if root.left is None and root.right is None:
            return [1, root.data, root.data, 1, True]
        l = story_teller.largestBSTBT(root.left)
        r = story_teller.largestBSTBT(root.right)
        ret = [0, 0, 0, 0, False]
        ret[0] = 1 + l[0] + r[0]
        if l[4] and r[4] and l[1] < root.data and r[2] > root.data:
            ret[2] = min(l[2], min(r[2], root.data))
            ret[1] = max(r[1], max(l[1], root.data))
            ret[3] = ret[0]
            ret[4] = True
        else:
            ret[3] = max(l[3], r[3])
            ret[4] = False
        return ret

    # ── from Contruct_BST.py ──────────────────────────────────────────────────
    def printPaths(root):
        """
        Print all root-to-leaf paths in a binary tree.

        Example:
            root = story_teller.newNode(10)
            root.left = story_teller.newNode(8)
            root.right = story_teller.newNode(2)
            story_teller.printPaths(root)

        :param root: binary tree root (story_teller.newNode)
        """
        story_teller._printPathsRec(root, [], 0)

    def _printPathsRec(root, path, pathLen):
        if root is None:
            return
        if len(path) > pathLen:
            path[pathLen] = root.data
        else:
            path.append(root.data)
        pathLen += 1
        if root.left is None and root.right is None:
            print(' '.join(str(x) for x in path[:pathLen]))
        else:
            story_teller._printPathsRec(root.left, path, pathLen)
            story_teller._printPathsRec(root.right, path, pathLen)

    # ── from goodness.py ──────────────────────────────────────────────────────
    def goodness(s):
        """
        Goodness = product(len of lowercase runs) - sum(len of digit runs).

        Example:
            story_teller.goodness("defGEhfX2")  # -> 5

        :param s: str
        :return: int
        """
        mul = reduce(operator.mul, (len(m) for m in re.findall(r'[a-z]+', s)), 1)
        sub = sum(len(m) for m in re.findall(r'[0-9]+', s))
        return mul - sub

    # ── from numberOfsequence.py ──────────────────────────────────────────────
    def prime_factors(n):
        """
        Return the prime factors of n.

        :param n: int
        :return: list of int
        """
        i = 2
        factors = []
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                factors.append(i)
        if n > 1:
            factors.append(n)
        return factors

    def primes(n):
        """
        Return a list of primes < n using a sieve.

        :param n: int
        :return: list of int
        """
        sieve = [True] * (n // 2)
        for i in range(3, int(n ** 0.5) + 1, 2):
            if sieve[i // 2]:
                sieve[i * i // 2::i] = [False] * ((n - i * i - 1) // (2 * i) + 1)
        return [2] + [2 * i + 1 for i in range(1, n // 2) if sieve[i]]

    def number_nice(A):
        """
        Count the number of "nice" sequences consistent with A (mod 10^9+7).
        https://www.hackerrank.com/contests/w22/challenges/number-of-sequences/

        A sequence is "nice" if for every position k (1-indexed):
          - 0 <= a[k] <= k-1
          - For every divisor pair (k|m): a[m] % k == a[k]

        Algorithm: sieve over prime powers p^e <= N.
          For each p^e, check all multiples in A. If any known values
          disagree mod p^e → contradiction (return 0). If all free (-1),
          multiply answer by p (each prime-power level adds exactly p choices).

        Time: O(N log N)  Space: O(N)

        Example:
            story_teller.number_nice([-1]*6)  # -> 60

        :param A: list of int (-1 means wildcard/free)
        :return: int (count mod 10^9+7)
        """
        MOD = 10 ** 9 + 7
        N = len(A)
        if A[0] != 0 and A[0] != -1:
            return 0
        nice_seqs = 1
        for prime in story_teller.primes(N + 1):
            e = 1
            while prime ** e <= N:
                q = prime ** e
                fixed_val = -1
                for k in range(q, N + 1, q):
                    if A[k - 1] != -1:
                        val = A[k - 1] % q
                        if fixed_val != -1 and fixed_val != val:
                            return 0   # contradiction
                        fixed_val = val
                if fixed_val == -1:
                    nice_seqs = (nice_seqs * prime) % MOD
                e += 1
        return nice_seqs

    # ── from roadLibrary.py ───────────────────────────────────────────────────
    def _DFSrec(adj, s, visited, val):
        visited[s] = 1
        val += 1
        for i in adj[s]:
            if visited[i] == 0:
                val = story_teller._DFSrec(adj, i, visited, val)
        return val

    def roadsAndLibraries(n, c_lib, c_road, cities):
        """
        Minimum cost to give every city access to a library.
        https://www.hackerrank.com/challenges/torque-and-development/problem

        Example:
            story_teller.roadsAndLibraries(3, 2, 1, [[1,2],[3,1],[2,3]])  # -> 4

        :param n: int - number of cities
        :param c_lib: int - cost to build a library
        :param c_road: int - cost to build a road
        :param cities: list of [int, int]
        :return: int - minimum total cost
        """
        if c_road > c_lib:
            return n * c_lib
        adj = {}
        for u, v in cities:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
        for i in range(1, n + 1):
            adj.setdefault(i, [])
        visited = [0] * (n + 1)
        components = []
        for i in range(1, n + 1):
            if visited[i] == 0:
                components.append(story_teller._DFSrec(adj, i, visited, 0))
        total = sum(c_road * (nodes - 1) for nodes in components)
        total += len(components) * c_lib
        return total

    # ── from shortestGraph.py ─────────────────────────────────────────────────
    def _bfs_weight(g, target_nodes, node, limit=-1):
        visited = set()
        q = Queue()
        q.put((node, 0))
        while not q.empty():
            n, w = q.get()
            if n in visited:
                continue
            if n in target_nodes and n != node:
                return w
            visited.add(n)
            if w == limit:
                return -1
            for nxt in g[n]:
                if nxt not in visited:
                    q.put((nxt, w + 1))
        return -1

    def findShortest(graph_nodes, graph_from, graph_to, ids, val):
        """
        Find shortest path between two nodes that share color val.

        Example:
            story_teller.findShortest(5, [1,1,2,3], [2,3,4,5], [1,2,3,3,2], 2)

        :param graph_nodes: int
        :param graph_from: list of int
        :param graph_to: list of int
        :param ids: list of int (color of each node, 1-indexed)
        :param val: int (target color)
        :return: int (shortest distance, -1 if impossible)
        """
        g = {i + 1: [] for i in range(graph_nodes)}
        for i in range(len(graph_from)):
            g[graph_from[i]].append(graph_to[i])
            g[graph_to[i]].append(graph_from[i])
        target_nodes = [i + 1 for i, c in enumerate(ids) if c == val]
        result = -1
        for node in target_nodes:
            w = story_teller._bfs_weight(g, target_nodes, node, result)
            if w > 0 and (w < result or result == -1):
                result = w
        return result

    # ── from temp_close_0.py ──────────────────────────────────────────────────
    def closestToZero(temps):
        """
        Return the temperature closest to zero (positive wins on tie).
        Returns 0 if list is empty.

        Example:
            story_teller.closestToZero([-2, 1, 3, 5])  # -> 1

        :param temps: list of int
        :return: int
        """
        closest = None
        for current in temps:
            if (closest is None
                    or abs(closest) > abs(current)
                    or (abs(closest) == abs(current) and closest < current)):
                closest = current
        return 0 if closest is None else closest

    # ── from shuffle_deck.py ──────────────────────────────────────────────────
    def shuffleDeck():
        """
        Build a standard 52-card deck and return it shuffled.

        Example:
            deck = story_teller.shuffleDeck()

        :return: list of namedtuple card(rank, suit)
        """
        suits = ['Spades', 'Diamonds', 'Hearts', 'Clubs']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        Card = namedtuple('card', ['suit', 'rank'])
        cards = [Card(rank, suit) for suit in suits for rank in ranks]
        random.shuffle(cards)
        return cards

class problem308:
    """
    # Determine the number of ways to group the array elements using parentheses so that the entire expression evaluates to True.
    #
    # For example, suppose the input is ['F', '|', 'T', '&', 'T']. In this case, there are two acceptable groupings: (F | T) & T and F | (T & T).
    """
    def split(expression):
        operands, operators = [], []

        for value in expression:
            if value in {'T', 'F'}:
                operands.append(value)
            else:
                operators.append(value)

        return operands, operators

    def solve(expression):
        operands, operators = problem308.split(expression)

        n = len(operands)
        T = [[0 for _ in range(n)] for _ in range(n)]
        F = [[0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            if operands[i] == 'T':
                T[i][i] = 1; F[i][i] = 0
            else:
                T[i][i] = 0; F[i][i] = 1

        for gap in range(1, n):
            for i in range(n - gap):
                j = i + gap

                for k in range(i, j):
                    all_options = (T[i][k] + F[i][k]) * (T[k+1][j] + F[k+1][j])

                    if operators[k] == '&':
                        T[i][j] += T[i][k] * T[k+1][j]
                        F[i][j] += (all_options - T[i][j])

                    elif operators[k] == '|':
                        F[i][j] += F[i][k] * F[k+1][j]
                        T[i][j] += (all_options - F[i][j])

                    elif operators[k] == '^':
                        T[i][j] += F[i][k] * T[k+1][j] + T[i][k] * F[k+1][j]
                        F[i][j] += T[i][k] * T[k+1][j] + F[i][k] * F[k+1][j]

        return T[0][n - 1]


class Node_tree:
    def __init__(self, val: int):
        self.val = val
        self.l = None
        self.r = None

    def __repr__(self):
        return "{}=[l->{}, r->{}]".format(self.val, self.l, self.r)


def make_cartree(arr: List[int], last: Node_tree, root: Node_tree):
    """cartree
    cartree = make_cartree([4,3,7,2,6,1,9], None, None)
    print(str(cartree) == '1=[l->2=[l->3=[l->4=[l->None, r->None], r->7=[l->None, r->None]], r->6=[l->None, r->None]], r->9=[l->None, r->None]]'

    cartree = make_cartree([3, 2, 6, 1, 9], None, None)
    assert str(cartree) == "1=[l->2=[l->3=[l->None, r->None], r->6=[l->None, r->None]], r->9=[l->None, r->None]]"

    :param arr:
    :param last:
    :param root:
    :return:
    """

    if not arr:
        return root

    node = Node_tree(arr[0])
    if not last:
        return make_cartree(arr[1:], node, node)

    if last.val > node.val:
        node.l = last
        return make_cartree(arr[1:], node, node)

    last.r = node
    return make_cartree(arr[1:], last, last)


class Player:
    """
        series play
    """
    def __init__(self, name, score):
        self.name = name
        self.score = score
    def __repr__(self):
        pass
    def comparator(a, b):
        val = b.score - a.score
        if val == 0:
            return -1 if a.name < b.name else 1
        return val

    def score_billboard(data):
        """
        Name - Score compare one another
        n = int(input())
        data = []
        for i in range(n):
            name, score = input().split()
            score = int(score)
            player = Player(name, score)
            data.append(player)
        Player.score_billboard(data)

        data append Player
        5
        amy 100
        david 100
        heraldo 50
        aakansha 75
        aleksa 150
        :return:

        data = [
            Player("amy", 100),
            Player("david", 100),
            Player("heraldo", 50),
            Player("aakansha", 75),
            Player("aleksa", 150)
        ]
        
        print("Running score_billboard:")
        Player.score_billboard(data)
        """
        data = sorted(data, key=cmp_to_key(Player.comparator))
        # data = sorted(data, key=lambda p: (-p.score, p.name)

        for i in data:
            print(i.name, i.score)


    @print_param("output_compareTriplets.txt", BASE_DIR)
    def compareTriplets(a, b):
        """
            compare_award
            a = list(map(int, input().rstrip().split()))
            b = list(map(int, input().rstrip().split()))

        :param a: list point a 17 28 30
        :param b: list point b 99 16 8
        :return: point for each player in each round
        """
        A = a
        B = b
        C = sum([1 if x[0] > x[1] else 0 for x in zip(A,B)])
        D = sum([1 if x[1] > x[0] else 0 for x in zip(A,B)])
        return [C, D]

    def climbingLeaderboard(scores, alice):
        """
        scores_count = int(input())

        scores = list(map(int, input().rstrip().split()))

        alice_count = int(input())

        alice = list(map(int, input().rstrip().split()))

        result = climbingLeaderboard(scores, alice)

        print(result)

    6
    100 90 90 80 75 60
    5
    50 65 90 77 102
        :param scores:
        :param alice:
        :return: array of rank
        """
        # List to contain and return Alice's ranks.
        results = []

        # Unique values from scores, since duplicate scores will have same rank (their index value).
        leaderboard = sorted(set(scores), reverse = True)

        # Use this var to track index within leaderboard later.
        l = len(leaderboard)

        # Loop through each of Alice's scores
        for a in alice:

            # If Alice's score is >= the score at the index of the end of leaderboard...
            # Subtract 1 from that index value (which is also the rank) to check the next score up.
            # If the score is less than the next score up, the index (rank) will be added to results.
            while (l > 0) and (a >= leaderboard[l-1]):
                l -= 1

            # We add 1 to the appended value to account for 0-indexing.
            results.append(l+1)

        return results


class Solution(object):
    def nextPermutation(self, nums):
        """
            #   arr = [0, 1, 0]
            #   next_permutation(arr)  (returns True)
            #   arr has been modified to be [1, 0, 0]
        https://www.nayuki.io/res/next-lexicographical-permutation-algorithm/nextperm.py
        :type nums: List[int]
        :rtype: list and arr modified to the next smallest permutation
        """
        arr = nums
        # Find non-increasing suffix
        i = len(arr) - 1
        while i > 0 and arr[i - 1] >= arr[i]:
            i -= 1
        if i <= 0:
            return False

        # Find successor to pivot
        j = len(arr) - 1
        while arr[j] <= arr[i - 1]:
            j -= 1
        arr[i - 1], arr[j] = arr[j], arr[i - 1]

        # Reverse suffix
        print(arr)
        arr[i:] = arr[len(arr) - 1 : i - 1 : -1]
        print(arr)
        return True

    def findJudge(self, N: int, trust: List[List[int]]) -> int:
        """
        Find the town judge.
        A town judge has egress 0 and ingress N-1.

        Example:
            Solution().findJudge(2, [[1, 2]])  # -> 2
        """
        ingress = defaultdict(set)
        egress = defaultdict(set)
        for p, q in trust:
            egress[p].add(q)
            ingress[q].add(p)
        for i in range(1, N+1):
            if len(egress[i]) == 0 and len(ingress[i]) == N - 1:
                return i
        return -1

class MySpecialQueue:
    """
    Special queue to get the highest bidder to maximize profit
    """
    def __init__(self):
        # Do not change the variable name of self.queue
        self.queue = None

    def insert(self, data):

        if self.queue == None:
            self.queue = [data]
        else:
            if data != '':
                self.queue.append(data)

    def dequeue(self):
        largest = max(self.queue)
        self.queue.remove(largest)

    def leftover_bidders( bids, number_of_items ) :
        """
        print(MySpecialQueue.leftover_bidders([1,2,3,4,5,6,7,8,9,], 2 ))

        :param bids:
        :param number_of_items:
        :return:
        """
        ######### DO NOT MODIFY BELOW ###########
        myQueue = MySpecialQueue()

        for bid in bids:
            myQueue.insert(bid)
        for sale in range(number_of_items):
            myQueue.dequeue()

        return myQueue.queue if myQueue.queue else [None]

        ######### DO NOT MODIFY ABOVE ###########

class Stock:
    def stock_gain():
        """
            the most gain
            # 7
            # 3 4 1 2 1 5 1
            :return:
        """
        n = int(input())
        prices = map(int, input().split())
        print(prices)
        gain = 0
        low = next(prices)
        for p in prices:
            low = min(low, p)
            gain = max(gain, p - low)
        print("gain",gain)

    pass


class Combo:
    """
        Daily Coding Problem: Problem #313
        open lock with 3 key revolt left or right
        solve by recursion and if possible try breath first search
        m = get_moves(start =Combo(0, 0, 0), target=Combo(4, 7, 6), deadends={Combo(6, 6, 6)})
        print(m)
    """
    def __init__(self, key_1: int, key_2: int, key_3: int):
        self.key_1 = key_1 if key_1 > -1 else key_1 + 10
        self.key_2 = key_2 if key_1 > -1 else key_1 + 10
        self.key_3 = key_3 if key_1 > -1 else key_1 + 10

    def __hash__(self):
        return hash((self.key_1, self.key_2, self.key_3))

    def __eq__(self, other):
        return \
            self.key_1 == other.key_1 and \
            self.key_2 == other.key_2 and \
            self.key_3 == other.key_3

    def __repr__(self):
        return "{}-{}-{}".format(self.key_1, self.key_2, self.key_3)


def get_moves(target: Combo, deadends: Set[Combo],
              start: Combo = Combo(0, 0, 0)):
    if start == target:
        return 0
    elif start in deadends:
        return None

    if start.key_1 != target.key_1:
        k1_moves = list()
        k1_diff = abs(start.key_1 - target.key_1)
        k1_new_start = Combo(target.key_1, start.key_2, start.key_3)
        k1_moves = [
            k1_diff + get_moves(target, deadends, k1_new_start),
            (10 - k1_diff) + get_moves(target, deadends, k1_new_start)
        ]
        k1_moves = [x for x in k1_moves if x]
        print('k1',k1_moves)
        if k1_moves:
            return min(k1_moves)

    if start.key_2 != target.key_2:
        k2_moves = list()
        k2_diff = abs(start.key_1 - target.key_1)
        k2_new_start = Combo(start.key_1, target.key_2, start.key_3)
        k2_moves = [
            k2_diff + get_moves(target, deadends, k2_new_start),
            (10 - k2_diff) + get_moves(target, deadends, k2_new_start)
        ]
        k2_moves = [x for x in k2_moves if x]
        print('k2',k2_moves)
        if k2_moves:
            return min(k2_moves)

    if start.key_2 != target.key_3:
        k3_moves = list()
        k3_diff = abs(start.key_1 - target.key_1)
        k3_new_start = Combo(start.key_1, start.key_2, target.key_3)
        k3_moves = [
            k3_diff + get_moves(target, deadends, k3_new_start),
            (10 - k3_diff) + get_moves(target, deadends, k3_new_start)
        ]
        k3_moves = [x for x in k3_moves if x]
        print('k3', k3_moves)
        if k3_moves:
            return min(k3_moves)

    return None

class greedy:
    def printMaxActivities(activities):
        """
            # Activity Selection Problem  in Python
            # greedy
    activities = [["A1", 0, 6],
                  ["A2", 3, 4],
                  ["A3", 1, 2],
                  ["A4", 5, 8],
                  ["A5", 5, 7],
                  ["A6", 8, 9]
                    ]
        greedy.printMaxActivities(activities)
        :param activities:
        :return:
        """
        activities.sort(key=lambda x: x[2])
        i = 0
        firstA = activities[i][0]
        print(firstA)
        for j in range(len(activities)):
            if activities[j][1] > activities[i][2]:
                print(activities[j][0])
                i = j

    def coinChange(totalNumber, coins):
        """
            Greedy
            greedy.coinChange(201, [1,2,5,20,50,100])

        :param coins: [1,2,5,20,50,100]
        :return:    100
                    100
                    1
        """
        N = totalNumber
        coins.sort()
        index = len(coins)-1
        while True:
            coinValue = coins[index]
            if N >= coinValue:
                N = N - coinValue
            if N < coinValue:
                index -= 1
            if N == 0:
                break


class DisjointSet:
# Kunno And Tree
# https://math.stackexchange.com/questions/838792/counting-triplets-with-red-edges-in-each-pair?newreg=60eee35f0b3844de852bda39f6dfec88
# https://www.hackerrank.com/contests/w5/challenges/kundu-and-tree
    def __init__(self, N):
        self.parent = [i for i in range(N)]
        self.total = [1] * N

    def union(self, a, b):
        a_parent = self.find(a)
        b_parent = self.find(b)
        if a_parent != b_parent:
            self.parent[b_parent] = a_parent
            self.total[a_parent] += self.total[b_parent]

    def find(self, a):
        if self.parent[a] != a:
            self.parent[a] = self.find(self.parent[a])
        return self.parent[a]

    def get_total(self, a):
        return self.total[self.find(a)]

if __name__ == '__main__':
    N = int(input())
    ds = DisjointSet(N)
    for i in range(N - 1):
        x, y, color = input().split()
        if color == 'b':
            ds.union(int(x) - 1, int(y) - 1)
    set_size = {ds.find(i): ds.get_total(i) for i in range(N)}
    complement = sum(x * (x - 1) * (N - x) // 2 +              #1
                     x * (x - 1) * (x - 2) // 6                #2
                     for x in set_size.values())
    print((N * (N - 1) * (N - 2) // 6 - complement) % (10 ** 9 + 7))


# super maximum cost query
# Complete the solve function below.

from bisect import bisect_right  # noqa: E402
parents = {}
rep = {}
def make_set(n):
    global parents,rep
    parents=dict(zip(range(1,n+1),range(1,n+1)))
    rep=dict(zip(range(1,n+1),({i} for i in range(1,n+1))))

def add_edge(x, y,paths,w):
    xroot = find(x)
    yroot = find(y)
    paths[w]+=len(rep[xroot])*len(rep[yroot])
    if xroot == yroot:
        return
    else:
        if len(rep[yroot])<len(rep[xroot]):
            parents[yroot] = xroot
            rep[xroot].update(rep[yroot])
            del rep[yroot]
        else:
            parents[xroot] = yroot
            rep[yroot].update(rep[xroot])
            del rep[xroot]

def find(x):
    if parents[x] != x:
        parent = find(parents[x])
        parents[x] = parent
    return parents[x]

@print_param("output_graph_supersum.txt", BASE_DIR)
def solve(tree, queries):
    """
        5 5
        1 2 3
        1 4 2
        2 5 6
        3 4 1
        1 1
        1 2
        2 3
        2 5
        1 6

    nq = input().split()

    n = int(nq[0])

    q = int(nq[1])

    tree = []

    for _ in range(n-1):
        tree.append(list(map(int, input().rstrip().split())))

    queries = []

    for _ in range(q):
        queries.append(list(map(int, input().rstrip().split())))

    result = solve(tree, queries)


#
# #Another solution to super
#
# # Complete the solve function below.
#
# #!/bin/python3
#
#
# class disjoint_set:
#     class Node:
#         def __init__(self, data = 0):
#             self.data = data
#             self.parent = self
#             self.rank = 0
#             self.size = 1
#
#     def __init__(self):
#         self.items = dict()
#         self.ans = 0
#
#     def make_set(self, data):
#         if not data in self.items:
#             self.items[data] = self.Node(data)
#         return self.items
#
#     def find_set(self, data):
#         if data in self.items:
#             node = self.items[data]
#         else:
#             return False
#
#         if node.parent == node:
#             return node
#         node.parent = self.find_set(node.parent.data)
#
#         return node.parent
#
#     def union(self, rep1, rep2):
#         node1 = self.find_set(rep1)
#         node2 = self.find_set(rep2)
#
#         #print("union: node1 = {} node2 = {}".format(node1.data, node2.data))
#
#         if node1 and node2 and node1 != node2:
#             if node1.rank >= node2.rank:
#                 if node1.rank == node2.rank:
#                     node1.rank += 1
#                 self.ans -= (node1.size*(node1.size - 1))//2 + (node2.size*(node2.size - 1))//2
#                 node2.parent = node1
#                 node1.size += node2.size
#                 self.ans += (node1.size*(node1.size - 1))//2
#             else:
#                 self.ans -= (node1.size*(node1.size - 1))//2 + (node2.size*(node2.size - 1))//2
#                 node1.parent = node2
#                 node2.size += node1.size
#                 self.ans += (node2.size*(node2.size - 1))//2
#         return True
#
#     def get_size(self, rep):
#         return self.find_set(rep).size
#
#     def get_ans(self):
#         return self.ans
#
# # Complete the solve function below.
# def solve(tree, queries):
#     dset = disjoint_set()
#     tree = sorted(tree, key=lambda x: x[2])
#     weights = list(map(lambda x: x[2], tree))
#     anses = []
#
#     for el in tree:
#         dset.make_set(el[0])
#         dset.make_set(el[1])
#         dset.union(el[0], el[1])
#
#         anses.append(dset.get_ans())
#         print("adding {} ans = {}".format(el, dset.get_ans()))
#
#     print("weights: {} anses: {}".format(weights, anses))
#     # do queries
#     output = []
#     for q in queries:
#         qleft, qright = q[0], q[1]
#
#         if qright < weights[0]:
#             output.append(0)
#         else:
#             right = bisect_right(weights, qright) - 1
#             print("query: {} RIGHT weights[{}] = {}".format(q, right, weights[right]))
#
#             if qleft <= weights[0]:
#                 output.append(anses[right])
#             else:
#                 left = bisect_left(weights, qleft) - 1
#                 print("query: {} LEFT weights[{}] = {}".format(q, left, weights[left]))
#                 output.append(anses[right] - anses[left])
#
#
#     return output
# #
# # if __name__ == '__main__':
# #     os.environ['HOME'] = '/Users/petern/Desktop/Python/DataStructure/graph_supersum.txt'
# #
# #     fptr = open(os.environ['HOME'], 'w')
# #
# #     nq = input().split()
# #
# #     n = int(nq[0])
# #
# #     q = int(nq[1])
# #
# #     tree = []
# #
# #     for _ in range(n-1):
# #         tree.append(list(map(int, input().rstrip().split())))
# #
# #     queries = []
# #
# #     for _ in range(q):
# #         queries.append(list(map(int, input().rstrip().split())))
# #
# #     result = solve(tree, queries)
# #
# #     fptr.write(str(result))
# #
# #     fptr.close()
# #
# #     myfile = open(os.environ['HOME'],'r')
# #
# #     print((myfile.readlines()))
#
    :param tree:
    :param queries:
    :return:
    """
    n = len(tree)+1
    tree.sort(key=lambda e:e[2])
    paths = {0:0}
    weights = [0]
    prev = 0
    make_set(len(tree)+1)
    for a,b,w in tree:
        if w != prev:
            weights.append(w)
            paths[w] = paths[prev]
        add_edge(a,b,paths,w)
        prev=w
    for l,r in queries:
        wr = weights[bisect_right(weights,r)-1]
        wl = weights[bisect_right(weights,l-1)-1]
        yield paths[wr]-paths[wl]

# An 8-puzzle is a game played on a 3 x 3 board of tiles, with the ninth tile missing.
# The remaining tiles are labeled 1 through 8 but shuffled randomly.
# Tiles may slide horizontally or vertically into an empty space, but may not be removed from the board.
#
# Design a class to represent the board, and find a series of steps
# to bring the board to the state [[1, 2, 3], [4, 5, 6], [7, 8, None]].
from copy import copy  # noqa: E402

class Board:
    def __init__(self, nums, goal='123456780'):
        self.goal = list(map(int, goal))
        self.tiles = list(map(int, nums))
        self.empty = self.tiles.index(0)
        self.original = copy(self.tiles)
        self.heuristic = self.heuristic()

    def __lt__(self, other):
        return self.heuristic < other.heuristic

    def manhattan(self, a, b):
        a_row, a_col = a // 3, a % 3
        b_row, b_col = b // 3, b % 3
        return abs(a_row - b_row) + abs(a_col - b_col)

    def heuristic(self):
        total = 0
        for digit in range(1, 9):
            total += self.manhattan(self.original.index(digit), self.tiles.index(digit))
            total += self.manhattan(self.tiles.index(digit), self.goal.index(digit))
        return total

    def swap(self, empty, diff):
        tiles = copy(self.tiles)
        tiles[empty], tiles[empty + diff] = tiles[empty + diff], tiles[empty]
        return tiles

    def get_moves(self):
        successors = []
        empty = self.empty

        if empty // 3 > 0:
            successors.append((Board(self.swap(empty, -3)), 'D'))
        if empty // 3 < 2:
            successors.append((Board(self.swap(empty, +3)), 'U'))
        if empty % 3 > 0:
            successors.append((Board(self.swap(empty, -1)), 'R'))
        if empty % 3 < 2:
            successors.append((Board(self.swap(empty, +1)), 'L'))

        return successors

    def search(start):
        heap = []
        closed = set()
        heapq.heappush(heap, [start.heuristic, 0, start, ''])

        while heap:
            _, moves, board, path = heapq.heappop(heap)
            if board.tiles == board.goal:
                return moves, path

            closed.add(tuple(board.tiles))
            for successor, direction in board.get_moves():
                if tuple(successor.tiles) not in closed:
                    item = [moves + 1 + successor.heuristic, moves + 1, successor, path + direction]
                    heapq.heappush(heap, item)

        return float('inf'), None

    def solve(nums):
        """
            nums = '124356870'
            print(Board.solve(nums))
        :param nums:
        :return:
        """
        start = Board(nums)
        count, path = Board.search(start)
        return count, path



class spiral:

    def shellcalc(n,s):
        if(n==1):
            return s
        elif(n==0):
            return 0
        else:
            sum=4*s+(n-1)*6
            s=s+4*(n-1)
            n=n-2
            return sum+spiral.shellcalc(n,s)

    def main(object):
        # call the function and print the value
        dim = int(input())
        diagonal = spiral.shellcalc(dim,1)
        print (diagonal)

def pickingNumbers(a):
    """
    Given an array of integers, find the longest subarray where the absolute difference between any two elements is less than or equal to 1
        n = int(input().strip())

        a = list(map(int, input().rstrip().split()))
    6
    4 6 5 3 3 1
    :param a:
    :return: [2, 1, 2, 3, 1]
    """
    # Write your code here
    #count the instances of the integer in the array
    from collections import Counter

    a= ["4","6","5","3","3","1"]
    count = Counter(a)
    #get the integer count and the integer + 1 count if exists, else 0
    #for the unique integers in the array

    all_combos = [(count.get(k) + count.get(str(int(k) + 1),0)) for k in count.keys()]
    print(count.keys())
    # now all we need is the max value of the combos,
    # keep in mind the combo can be just one integer!
    print(all_combos)
    return max(all_combos)

def hourglassSum(arr):
    """
    for _ in range(6):
            arr.append(list(map(int, input().rstrip().split())))

    result = hourglassSum(arr)
    print(result)

# 1 2 3 0 0 0
# 0 1 0 0 0 0
# 1 1 1 0 0 0
# 0 0 2 4 4 0
# 0 0 0 2 0 0
# 0 0 1 2 4 0
    :param arr:
    :return:
    """
    a = max([sum(arr[j][i:i+3]) + arr[j+1][i+1] + sum(arr[j+2][i:i+3])
             for j in range(len(arr)-2) for i in range(len(arr[0])-1)] )
    return a

# ── from BFShortest.py ────────────────────────────────────────────────────────
class Graph:
    """
    Find all distances from a starting node via BFS (each edge weight = 6).

    Example:
        g = Graph(4)
        g.connect(0, 1)
        g.connect(0, 2)
        g.find_all_distances(0)
    """
    def __init__(self, n):
        self.n = n
        self.edges = defaultdict(list)

    def connect(self, x, y):
        self.edges[x].append(y)
        self.edges[y].append(x)

    def find_all_distances(self, root):
        distances = [-1] * self.n
        unvisited = set(range(self.n))
        q = queue.Queue()
        distances[root] = 0
        unvisited.remove(root)
        q.put(root)
        while not q.empty():
            node = q.get()
            for child in self.edges[node]:
                if child in unvisited:
                    distances[child] = distances[node] + 6
                    unvisited.remove(child)
                    q.put(child)
        distances.pop(root)
        print(' '.join(map(str, distances)))


# ── from shuffle_deck.py ──────────────────────────────────────────────────────
class PlayingDeck:
    """
    A full 52-card playing deck with index-based access.

    Example:
        deck = PlayingDeck()
        random.shuffle(deck)
        for card in deck:
            print(card)
    """
    def __init__(self):
        suits = ['Spades', 'Diamonds', 'Hearts', 'Clubs']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        Card = namedtuple('card', ['suit', 'rank'])
        self.cards = [Card(rank, suit) for suit in suits for rank in ranks]

    def __len__(self):
        return len(self.cards)

    def __getitem__(self, position):
        return self.cards[position]

    def __setitem__(self, position, card):
        self.cards[position] = card


# =============================================================================
# ── CONSOLIDATED FROM vin.py ─────────────────────────────────────────────────
# =============================================================================

class ArrayProblems:
    """
    ArrayProblems
    =============
    Theory: A *sorted* array allows O(N) algorithms for statistics like mode,
    because equal elements are always adjacent. An unsorted array requires an
    O(N²) nested scan or an O(N) hash-map approach.

    Problems covered:
      1. mode_sorted   — Mode of a sorted array, O(N) time O(1) space
      2. mode_unsorted — Mode via nested scan, O(N²), for comparison only
    """

    def mode_sorted(arr):
        """
        Find the mode (most frequent element) of a *sorted* array in O(N).

        Example:
            ArrayProblems.mode_sorted([1,2,3,3,3,4,4,5,5,6])  # -> 3

        :param arr: sorted list of int
        :return: modal value (int)
        """
        count, best_count = 1, 1
        temp_mode, mode = arr[0], arr[0]
        for i in range(1, len(arr)):
            if temp_mode == arr[i]:
                count += 1
                if count > best_count:
                    best_count = count
                    mode = temp_mode
            else:
                temp_mode = arr[i]
                count = 1
        return mode

    def mode_unsorted(arr):
        """
        Find the mode via nested scan — O(N²), for educational comparison only.
        Prefer mode_sorted on sorted data, or Counter for unsorted data.

        Example:
            ArrayProblems.mode_unsorted([3,1,3,2,3])  # -> 3

        :param arr: list of int (any order)
        :return: modal value (int)
        """
        best_count, mode = 0, arr[0]
        for i in range(len(arr)):
            count = sum(1 for j in range(len(arr)) if arr[j] == arr[i])
            if count > best_count:
                best_count = count
                mode = arr[i]
        return mode

    @staticmethod
    def RightRotate(a, n, k):
        """
        Right rotate array elements by k positions and print.

        Example:
            ArrayProblems.RightRotate([1, 2, 3, 4, 5], 5, 2)
        """
        k = k % n
        for i in range(0, n):
            if i < k:
                print(a[n + i - k], end=" ")
            else:
                print(a[i - k], end=" ")
        print("\n")


class Stock:
    """
    Stock
    =====
    Theory: Stock problems are a classic application of dynamic programming.
      • 1 transaction: track running minimum → O(N) time O(1) space
      • K transactions: DP table prev/curr over k rounds → O(kN) time O(N) space

    Problems covered:
      1. stock_gain              — Best single buy/sell, O(N)
      2. max_profit_k_transactions — Best profit with at most k transactions, O(kN)
    """

    def stock_gain():
        """
        Find the maximum single-transaction gain from stdin.
        Reads n then space-separated prices.

        Example (stdin):
            7
            3 4 1 2 1 5 1
        -> gain 4
        """
        n = int(input())
        prices = map(int, input().split())
        gain = 0
        low = next(prices)
        for p in prices:
            low = min(low, p)
            gain = max(gain, p - low)
        print("gain", gain)

    def max_profit_k_transactions(prices, k):
        """
        Maximum profit from at most k buy/sell transactions (must sell before
        buying again).

        Algorithm: DP — for each transaction t, iterate prices maintaining
          max_diff = max(prev_trans[i-1] - prices[i-1])
        Time: O(k*N)  Space: O(N)

        Example:
            Stock.max_profit_k_transactions([2,4,1,7], 2)  # -> 8

        :param prices: list of int
        :param k: int, max number of transactions
        :return: int, maximum profit
        """
        prev = [0] * len(prices)
        curr = []
        for _ in range(k):
            curr = [0]
            max_diff = float('-inf')
            for i in range(1, len(prices)):
                max_diff = max(max_diff, prev[i - 1] - prices[i - 1])
                curr.append(max(curr[i - 1], prices[i] + max_diff))
            prev = curr[:]
        return curr[-1] if curr else 0


class FibonacciShowcase:
    """
    FibonacciShowcase
    =================
    Theory: F(N) = F(N-1) + F(N-2), with F(1) = F(2) = 1.

    DECISION GUIDE — when to use which:
    ┌─────────────────────────────────┬────────────┬──────────┬────────────────────────────────┐
    │ Method                          │ Time       │ Space    │ Best Use Case                  │
    ├─────────────────────────────────┼────────────┼──────────┼────────────────────────────────┤
    │ fibo         (iterative)        │ O(N)       │ O(1)     │ DEFAULT: single query, N<10^7  │
    │ bin_fibo     (fast doubling)    │ O(log N)   │ O(log N) │ Very large N (>10^6), mod ops  │
    │ dyna_fibo    (top-down memo)    │ O(N)       │ O(N)     │ Many repeated queries          │
    │ dyna_fibo2   (top-down array)   │ O(N)       │ O(N)     │ Alternative top-down           │
    │ fibonacci_bu (bottom-up cache)  │ O(N)       │ O(N)     │ Incremental queries            │
    │ simple_fibo  (tail recursion)   │ O(N)       │ O(N)     │ Teaching / elegant recursion   │
    │ my_fib       (dict memoization) │ O(N)       │ O(N)     │ Teaching top-down memoization  │
    └─────────────────────────────────┴────────────┴──────────┴────────────────────────────────┘

    WHY NOT SIEVE FOR FIBONACCI?
    The Sieve of Eratosthenes marks multiples of primes — it is for finding
    *prime numbers*, not for computing sequences defined by recurrence.
    Fibonacci is defined by F(n) = F(n-1) + F(n-2), not by divisibility.
    Use the sieve when the problem involves primality or divisibility constraints
    (e.g., story_teller.primes, story_teller.number_nice). Use fibo/bin_fibo
    when you need the N-th term of the Fibonacci sequence.

    All methods use F(1)=F(2)=1 (1-indexed, same base).
    """

    def fibo(N):
        """
        Iterative Fibonacci. O(N) time, O(1) space.
        RECOMMENDED for single queries.

        Example:
            FibonacciShowcase.fibo(10)  # -> 55
        """
        a = b = 1
        for _ in range(2, N):
            a, b = b, a + b
        return b

    def simple_fibo(N, a=0, b=1):
        """
        Tail-recursive Fibonacci. O(N) time, O(N) stack space.
        Elegant but hits Python's recursion limit for large N.

        Example:
            FibonacciShowcase.simple_fibo(10)  # -> 55
        """
        if N < 3:
            return a + b
        return FibonacciShowcase.simple_fibo(N - 1, b, a + b)

    def dyna_fibo(N, memo=None):
        """
        Top-down memoized Fibonacci (dict). O(N) time, O(N) space.
        Pass a fresh dict each call to avoid cross-call contamination.

        Example:
            FibonacciShowcase.dyna_fibo(10, {1:1, 2:1})  # -> 55
        """
        if memo is None:
            memo = {1: 1, 2: 1}
        if N not in memo:
            memo[N] = FibonacciShowcase.dyna_fibo(N - 1, memo) + \
                      FibonacciShowcase.dyna_fibo(N - 2, memo)
        return memo[N]

    def dyna_fibo2(N, memo=None):
        """
        Top-down memoized Fibonacci (list). O(N) time, O(N) space.
        Uses a list instead of dict — marginally faster indexing.

        Example:
            FibonacciShowcase.dyna_fibo2(10)  # -> 55
        """
        if memo is None:
            memo = [0, 1, 1] + [0] * N
        if not memo[N]:
            memo[N] = FibonacciShowcase.dyna_fibo2(N - 1, memo) + \
                      FibonacciShowcase.dyna_fibo2(N - 2, memo)
        return memo[N]

    def fibonacci_bu(N):
        """
        Bottom-up (tabulation) Fibonacci. O(N) time, O(N) space.
        Builds the full cache iteratively — good for incremental queries.
        F(1)=1, F(2)=1, F(3)=2, ...

        Example:
            FibonacciShowcase.fibonacci_bu(10)  # -> 55
        """
        if N <= 2:
            return 1
        a, b = 1, 1
        for _ in range(2, N):
            a, b = b, a + b
        return b

    def bin_fibo(N):
        """
        Fast-doubling (binary) Fibonacci. O(log N) time, O(log N) space.
        Best for very large N; uses matrix identity:
          F(2k)   = F(k) * (2*F(k+1) - F(k))
          F(2k+1) = F(k)^2 + F(k+1)^2

        Example:
            FibonacciShowcase.bin_fibo(10)  # -> 55
        """
        a, b = 0, 1
        f0, f1 = 1, 1
        r, s = (1, 1) if N & 1 else (0, 1)
        N //= 2
        while N > 0:
            a, b = f0 * a + f1 * b, f0 * b + f1 * (a + b)
            f0, f1 = b - a, a
            if N & 1:
                r, s = f0 * r + f1 * s, f0 * s + f1 * (r + s)
            N //= 2
        return r

    def my_fib(N, memo=None):
        """
        Top-down memoized Fibonacci (dict, explicit None guard).
        Teaching version — shows the memoization pattern clearly.

        Example:
            FibonacciShowcase.my_fib(10)  # -> 55
        """
        if memo is None:
            memo = {}
        if memo.get(N):
            return memo[N]
        if N == 1 or N == 2:
            result = 1
        else:
            result = FibonacciShowcase.my_fib(N - 1, memo) + \
                     FibonacciShowcase.my_fib(N - 2, memo)
        memo[N] = result
        return result

    @staticmethod
    def fib_dn(N, memo=None):
        """
        Alternative top-down Fibonacci with default dict initialization.

        Example:
            FibonacciShowcase.fib_dn(10, {0:1, 1:1})  # -> 89
        """
        if memo is None:
            memo = {0: 1, 1: 1}
        if N not in memo:
            memo[N] = FibonacciShowcase.fib_dn(N-1, memo) + FibonacciShowcase.fib_dn(N-2, memo)
        return memo[N]

    def compare(N=990, count=100):
        """
        Benchmark all Fibonacci methods for F(N), repeated `count` times.
        Prints a ranked table of wall-clock times.

        Example:
            FibonacciShowcase.compare(N=990, count=100)
        """
        from timeit import timeit
        import sys

        methods = [
            ("fibo (iterative O(N) O(1))",
             lambda: FibonacciShowcase.fibo(N)),
            ("bin_fibo (fast-doubling O(logN))",
             lambda: FibonacciShowcase.bin_fibo(N)),
            ("dyna_fibo (top-down dict O(N))",
             lambda: FibonacciShowcase.dyna_fibo(N, {1: 1, 2: 1})),
            ("dyna_fibo2 (top-down list O(N))",
             lambda: FibonacciShowcase.dyna_fibo2(N)),
            ("fibonacci_bu (bottom-up O(N))",
             lambda: FibonacciShowcase.fibonacci_bu(N)),
            ("my_fib (dict memo O(N))",
             lambda: FibonacciShowcase.my_fib(N, {})),
            ("simple_fibo (tail-recursion O(N))",
             None),  # may hit recursion limit
        ]

        print(f"\n{'='*64}")
        print(f"  Fibonacci Benchmark: F({N}), {count} repetitions each")
        print(f"{'='*64}")
        print(f"  {'Method':<38} {'Time (s)':>10}  {'Rank':>5}")
        print(f"  {'-'*55}")

        results = []
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, N * 3))
        for name, fn in methods:
            if fn is None:
                try:
                    t = timeit(lambda: FibonacciShowcase.simple_fibo(N),
                               number=count)
                    results.append((name, t))
                except RecursionError:
                    results.append((name, float('inf')))
            else:
                t = timeit(fn, number=count)
                results.append((name, t))
        sys.setrecursionlimit(old_limit)

        results.sort(key=lambda x: x[1])
        for rank, (name, t) in enumerate(results, 1):
            time_str = f"{t:.6f}" if t != float('inf') else "RecursionError"
            print(f"  {name:<38} {time_str:>10}  #{rank}")
        print(f"{'='*64}\n")


class DynamicProgramming:
    """
    DynamicProgramming
    ==================
    Theory: Dynamic programming (DP) solves problems by breaking them into
    overlapping sub-problems and storing results to avoid recomputation.

    Two styles:
      • Top-down (memoization):  recursion + cache, natural problem decomposition
      • Bottom-up (tabulation):  fill a table iteratively, no stack overhead

    Problems covered:
      1. divide_numbers — Partition array into two subsets, minimise difference
                          (greedy recursive, exponential worst-case)
      2. lis_recursive  — Longest Increasing Subsequence via recursion, O(2^N)
                          [for comparison / teaching only]
      3. lis            — LIS via patience-sort / binary search, O(N log N)
                          [canonical best solution]
    """

    # ── Subset Partition ─────────────────────────────────────────────────────

    def _get_diff(s1, s1_sum, s2, s2_sum, score):
        """Recursive helper: greedily move the element that most reduces diff."""
        min_diff, min_cand = score, None
        for i, num in enumerate(s1):
            new_s1_sum = s1_sum - num
            new_s2_sum = s2_sum + num
            new_score = abs(new_s1_sum - new_s2_sum)
            if new_score < min_diff:
                min_diff = new_score
                min_cand = (s1[:i] + s1[i + 1:], new_s1_sum,
                            s2 + [num], new_s2_sum)
        if not min_cand:
            return set(s1), set(s2)
        return DynamicProgramming._get_diff(
            min_cand[0], min_cand[1], min_cand[2], min_cand[3], min_diff)

    def divide_numbers(nums):
        """
        Divide nums into two subsets with minimal absolute difference of sums.

        Example:
            DynamicProgramming.divide_numbers([5, 10, 15, 20, 25])
            # -> ({5, 15, 20}, {10, 25})  difference = 5

        :param nums: list of positive int
        :return: tuple of two sets
        """
        total = sum(nums)
        return DynamicProgramming._get_diff(nums[:], total, [], 0, total)

    # ── Longest Increasing Subsequence ────────────────────────────────────────

    def _lis_recursive_helper(arr, n):
        """Internal recursive LIS — O(2^N), teaching use only."""
        if n == 1:
            return 1
        max_here = 1
        for i in range(1, n):
            res = DynamicProgramming._lis_recursive_helper(arr, i)
            if arr[i - 1] < arr[n - 1] and res + 1 > max_here:
                max_here = res + 1
        return max_here

    def lis_recursive(arr):
        """
        Longest Increasing Subsequence length via naive recursion.
        Time: O(2^N) — teaching/comparison only.

        Example:
            DynamicProgramming.lis_recursive([1, 2, 5, 4, 6, 2])  # -> 4
        """
        return max(
            DynamicProgramming._lis_recursive_helper(arr, i)
            for i in range(1, len(arr) + 1)
        )

    def lis(nums):
        """
        Longest Increasing Subsequence via patience sort (binary search).
        Time: O(N log N)  Space: O(N)  ← canonical best solution

        Algorithm: maintain a virtual pile list where each pile's top is the
        smallest tail of all increasing subsequences of that length. Use binary
        search to find the correct pile for each element.

        Example:
            DynamicProgramming.lis([1, 2, 5, 4, 6, 2])  # -> 4
            DynamicProgramming.lis([10, 9, 2, 5, 3, 7, 101, 18])  # -> 4

        :param nums: list of int
        :return: int, length of LIS
        """
        tails = []

        def insert(target):
            lo, hi = 0, len(tails) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if tails[mid] >= target:
                    hi = mid - 1
                else:
                    lo = mid + 1
            if lo == len(tails):
                tails.append(target)
            else:
                tails[lo] = target

        for num in nums:
            insert(num)
        return len(tails)


class SearchAndCount:
    """
    SearchAndCount
    ==============
    Theory:
      • Binary search reduces search on sorted data from O(N) to O(log N).
      • Counter (hash map) reduces frequency queries to O(1) lookup after O(N) build.
      • Combining binary search + hash map allows O(M log N) pair-counting.

    Problems covered:
      1. binary_search  — Classic O(log N) search in sorted array
      2. count_pairs    — Count pairs summing to X across two sorted arrays, O(M log N)
      3. count_pairs3   — Count triplets (a+b==c) across three arrays, O(N²) with Counter
    """

    def binary_search(arr, value, low=None, high=None):
        """
        Search for value in sorted arr. Returns True if found.

        Example:
            SearchAndCount.binary_search([1,3,5,7,9], 5)  # -> True
            SearchAndCount.binary_search([1,3,5,7,9], 4)  # -> False
        """
        if low is None:
            low = 0
        if high is None:
            high = len(arr) - 1
        while low <= high:
            mid = (low + high) // 2
            if arr[mid] == value:
                return True
            elif arr[mid] > value:
                high = mid - 1
            else:
                low = mid + 1
        return False

    def count_pairs(arr1, arr2, x):
        """
        Count pairs (a, b) where a ∈ arr1, b ∈ arr2 and a+b == x.
        Both arrays must be sorted. Time: O(M log N).

        Example:
            SearchAndCount.count_pairs([1,3,5], [2,4,6], 7)  # -> 2

        :param arr1: sorted list of int
        :param arr2: sorted list of int
        :param x: target sum
        :return: int
        """
        count = 0
        n = len(arr2)
        for a in arr1:
            if SearchAndCount.binary_search(arr2, x - a, 0, n - 1):
                count += 1
        return count

    def count_pairs3(arr1, arr2, arr3):
        """
        Count triplets (a, b, c) where a ∈ arr1, b ∈ arr2, c ∈ arr3
        and a + b == c. Time: O(N²) using Counter for O(1) lookup.

        Example:
            SearchAndCount.count_pairs3([1,2], [3,4], [4,5,6])  # -> 2

        :param arr1: list of int
        :param arr2: list of int
        :param arr3: list of int (target sums)
        :return: int
        """
        count = 0
        c3 = Counter(arr3)
        c2 = Counter(arr2)
        c1 = Counter(arr1)
        for c_val, c_cnt in c3.items():
            for b_val, b_cnt in c2.items():
                complement = c_val - b_val
                if complement in c1:
                    count += c1[complement] * c_cnt * b_cnt
        return count


class GeometryProblems:
    """
    GeometryProblems
    ================
    Theory: Rectangle intersection can be solved in O(1) by projecting onto
    each axis independently. The x-overlap is [max(left1, left2), min(right1, right2)]
    and similarly for y. If either overlap is negative, there is no intersection.

    Problems covered:
      1. rectangle_intersection — Area of intersection of two axis-aligned rectangles
    """

    def rectangle_intersection(rec1, rec2):
        """
        Return the area of intersection of two axis-aligned rectangles.
        Each rectangle is a dict with keys 'top_left' (x,y) and
        'dimensions' (width, height). Returns 0 if they do not intersect.

        Example:
            r1 = {"top_left": (1, 4), "dimensions": (3, 3)}
            r2 = {"top_left": (0, 5), "dimensions": (4, 3)}
            GeometryProblems.rectangle_intersection(r1, r2)  # -> 6

        :param rec1: dict
        :param rec2: dict
        :return: int, area of intersection
        """
        left_x  = max(rec1["top_left"][0], rec2["top_left"][0])
        right_x = min(rec1["top_left"][0] + rec1["dimensions"][0],
                      rec2["top_left"][0] + rec2["dimensions"][0])
        top_y    = min(rec1["top_left"][1], rec2["top_left"][1])
        bottom_y = max(rec1["top_left"][1] - rec1["dimensions"][1],
                       rec2["top_left"][1] - rec2["dimensions"][1])
        if left_x > right_x or bottom_y > top_y:
            return 0
        return (right_x - left_x) * (top_y - bottom_y)


# =============================================================================
# ── CONSOLIDATED FROM Note.py ────────────────────────────────────────────────
# =============================================================================

class ClassicPuzzles:
    """
    ClassicPuzzles
    ==============
    Documented solutions for common interview puzzles.
    Each method includes the problem statement and an explanation.
    Uncomment the __main__ calls below each class to run examples.

    Puzzles:
      1. find_ith_digit        — Infinite sequence {1,2,2,3,3,3,...}: find i-th element
      2. last_unique_character — Last character that appears exactly once in a string
      3. highest_sum_pyramid   — Max path sum traversing a number pyramid top-to-bottom
      4. count_matrix_range    — Count elements in a sorted matrix smaller than M[i1,j1]
                                 and greater than M[i2,j2]
    """

    def find_ith_digit(i):
        """
        In the infinite sequence {1,2,2,3,3,3,4,4,4,4,...} (number k appears k times),
        return the i-th element (1-indexed).

        Algorithm: binary search for the group g such that g*(g-1)//2 < i <= g*(g+1)//2.
        Time: O(sqrt(i))  Space: O(1)

        Example:
            ClassicPuzzles.find_ith_digit(1)   # -> 1
            ClassicPuzzles.find_ith_digit(6)   # -> 3
            ClassicPuzzles.find_ith_digit(10)  # -> 4
        """
        group = 1
        while i > group * (group + 1) // 2:
            group += 1
        return group

    def last_unique_character(s):
        """
        Return the last character in s that appears exactly once,
        or 'none' if no such character exists.

        Algorithm: build frequency map, then scan string in reverse.
        Time: O(N)  Space: O(1) (bounded alphabet)

        Example:
            ClassicPuzzles.last_unique_character("slideeducation")  # -> 'u'
            ClassicPuzzles.last_unique_character("aabb")            # -> 'none'
        """
        freq = {}
        for ch in s:
            freq[ch] = freq.get(ch, 0) + 1
        for ch in reversed(s):
            if freq[ch] == 1:
                return ch
        return "none"

    def highest_sum_pyramid(pyramid):
        """
        Find the maximum sum path from top to bottom of a number pyramid,
        where each step moves to an adjacent element in the next row.

        Algorithm: bottom-up DP — each cell stores the best sum reachable
        from that position downward.
        Time: O(N²)  Space: O(N²)

        Example:
            ClassicPuzzles.highest_sum_pyramid([
                [3],
                [7, 4],
                [2, 4, 6],
                [8, 5, 9, 3]
            ])  # -> 23  (3->7->4->9)
        """
        rows = len(pyramid)
        memo = [row[:] for row in pyramid]
        for row in range(rows - 2, -1, -1):
            for col in range(row + 1):
                memo[row][col] = (pyramid[row][col] +
                                  max(memo[row + 1][col], memo[row + 1][col + 1]))
        return memo[0][0]

    def count_matrix_range(mat, i1, j1, i2, j2):
        """
        Given a matrix where every row and column is sorted, count elements
        that are strictly less than mat[i1][j1] OR strictly greater than mat[i2][j2].

        Time: O(N²)  Space: O(1)

        Example:
            mat = [
                [1,  3,  7, 10, 15, 20],
                [2,  6,  9, 14, 22, 25],
                [3,  8, 10, 15, 25, 30],
                [10, 11, 12, 23, 30, 35],
                [20, 25, 30, 35, 40, 45],
            ]
            ClassicPuzzles.count_matrix_range(mat, 1, 1, 3, 3)  # -> 15
        """
        lo, hi = mat[i1][j1], mat[i2][j2]
        return sum(1 for row in mat for x in row if x < lo or x > hi)


class PythonPatterns:
    """
    PythonPatterns
    ==============
    Reference examples for Python built-ins and idioms.
    All code is executable — uncomment the print/assert lines to run.

    Topics:
      1. Regex patterns — re.match, re.search, re.findall with cheat-sheet
      2. itertools.chain — Flatten iterables and use generator-based chaining

    Regex Quick-Guide (from Note.py):
      .        Any character except newline
      ^        Start of string (or line in MULTILINE)
      $        End of string (or line in MULTILINE)
      *        0 or more of preceding
      +        1 or more of preceding
      ?        0 or 1 of preceding
      {m,n}    Between m and n repetitions
      [abc]    Character class
      [^abc]   Negated character class
      (...)    Group
      \\.       Literal dot
      \\d        Digit [0-9]
      \\w        Word char [a-zA-Z0-9_]
      \\s        Whitespace
      \\b        Word boundary
      Flags: re.I (ignore case), re.M (multiline), re.S (dot matches newline)
    """

    def regex_examples():
        """
        Demonstrate key regex patterns.
        Each example is self-contained and prints its result.
        """
        import re

        s = 'vuthanhtai'

        # Match lowercase-only string
        m = re.match(r'[a-z]+', s)
        assert m is not None, "Expected match for all-lowercase string"

        # Search for substring 'thanh'
        m = re.match(r'(.*)(thanh)(.*)', s)
        assert m is not None

        # Case-insensitive match for leading 'V'
        m = re.match(r'V', s, re.IGNORECASE)
        assert m is not None

        # findall: extract integers from text
        text = "The answer is 42, not 0 or 100"
        nums = re.findall(r'[0-9]+', text)
        assert nums == ['42', '0', '100']

        # findall: extract tagged content (@ is NOT a word char, so <\w+> won't match <@tag>)
        # Use [@\w]+ to include @ in tag names
        html = "<red>AB CD<@red><yellow>EFGH<@yellow>"
        tags = re.findall(r'<[@\w]+>[\w ]*<[@\w]+>', html)
        assert tags == ['<red>AB CD<@red>', '<yellow>EFGH<@yellow>'], f"got {tags}"

        print("PythonPatterns.regex_examples: all assertions passed")

    def chain_examples():
        """
        Demonstrate itertools.chain for flattening iterables.

        chain(a, b, c)               — concatenate separate iterables
        chain.from_iterable(nested)  — flatten a list-of-lists (lazy)
        chain.from_iterable(gen)     — works with any generator
        """
        from itertools import chain

        list1, list2, list3 = ["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]

        # Direct chaining
        result = list(chain(list1, list2, list3))
        assert result == list("abcdefghi")

        # From iterable of iterables
        iterables = [list1, list2, list3]
        result2 = list(chain.from_iterable(iterables))
        assert result2 == result

        # Generator-based: produces [0], [0,1], [0,1,2], ... ranges
        def gen_iterables():
            for i in range(10):
                yield range(i)

        flat = list(chain.from_iterable(gen_iterables()))
        assert flat == [0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4,
                        0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6,
                        0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        print("PythonPatterns.chain_examples: all assertions passed")


# =============================================================================
# ── CONSOLIDATED FROM vin.py (ADDITIONAL & COMMENTED PROBLEMS) ───────────────
# =============================================================================

class PrimeSieveShowcase:
    """
    PrimeSieveShowcase
    ==================
    Theory and implementations of primality testing and sieve methods.
    """

    @staticmethod
    def is_prime(x):
        """
        Check if a number x is prime.
        Note: Prints value x if a factor is not found on the first check.

        Example:
            PrimeSieveShowcase.is_prime(991)  # -> True
        """
        for i in range(2, int(x**0.5) + 1):
            if x % i == 0:
                return False
            else:
                print(x)
                return True
        return x > 1

    @staticmethod
    def prime_interval(lower=900, upper=1000):
        """
        Print all prime numbers within the interval [lower, upper].

        Example:
            PrimeSieveShowcase.prime_interval(900, 1000)
        """
        print("Prime numbers between", lower, "and", upper, "are:")
        primes_found = []
        for num in range(lower, upper + 1):
            if num > 1:
                for i in range(2, num):
                    if (num % i) == 0:
                        break
                else:
                    print(num)
                    primes_found.append(num)
        return primes_found

    @staticmethod
    def simpleSieve(limit, primes_list):
        """
        Find all primes smaller than or equal to limit using Sieve of Eratosthenes.
        """
        mark = [False] * (limit + 1)
        for i in range(2, limit + 1):
            if not mark[i]:
                primes_list.append(i)
                for j in range(i, limit + 1, i):
                    mark[j] = True

    @staticmethod
    def primesInRange(low, high):
        """
        Print all prime numbers in the range [low, high] using Segmented Sieve.

        Example:
            PrimeSieveShowcase.primesInRange(10, 100)
        """
        from math import floor, sqrt
        limit = floor(sqrt(high)) + 1
        primes_list = list()
        PrimeSieveShowcase.simpleSieve(limit, primes_list)

        n = high - low + 1
        mark = [False] * (n + 1)

        for i in range(len(primes_list)):
            loLim = floor(low / primes_list[i]) * primes_list[i]
            if loLim < low:
                loLim += primes_list[i]
            if loLim == primes_list[i]:
                loLim += primes_list[i]

            for j in range(loLim, high + 1, primes_list[i]):
                mark[j - low] = True

        primes_found = []
        for i in range(low, high + 1):
            if not mark[i - low]:
                print(i, end=" ")
                primes_found.append(i)
        print()
        return primes_found


class NelderMeadShowcase:
    """
    NelderMeadShowcase
    ==================
    Multimodal function optimization using scipy.optimize.minimize (Nelder-Mead).
    """

    @staticmethod
    def objective(v):
        from numpy import exp, sqrt, cos, e, pi
        x, y = v
        return -20.0 * exp(-0.2 * sqrt(0.5 * (x ** 2 + y ** 2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

    @staticmethod
    def optimize():
        from scipy.optimize import minimize
        from numpy.random import rand
        r_min, r_max = -5.0, 5.0
        pt = r_min + rand(2) * (r_max - r_min)
        result = minimize(NelderMeadShowcase.objective, pt, method='nelder-mead')
        print('Status :', result['message'])
        print('Total Evaluations:', result['nfev'])
        print('Solution: f(%s) = %.5f' % (result['x'], NelderMeadShowcase.objective(result['x'])))
        return result


class PowerSumNTT:
    """
    PowerSumNTT
    ===========
    Implementation of the Power Sum Problem using Number Theoretic Transform (NTT).
    """
    mod = 998244353

    @staticmethod
    def power(n, k):
        mod = PowerSumNTT.mod
        if k == 0:
            return 1
        x = 1
        while k > 1:
            if k % 2 == 0:
                n = n * n % mod
            else:
                x = n * x % mod
                n = n * n % mod
            k //= 2
        return n * x % mod

    @staticmethod
    def get_pow2(k):
        pow2 = 1
        logpow2 = 0
        while pow2 <= k:
            pow2 *= 2
            logpow2 += 1
        invpow2 = PowerSumNTT.power(pow2, PowerSumNTT.mod - 2)
        return pow2, logpow2, invpow2

    @staticmethod
    def NTT(p, r, pow2, logpow2, invpow2, forward):
        mod = PowerSumNTT.mod
        j = pow2 // 2
        for i in range(1, pow2 - 1):
            if i >= j:
                p[i], p[j] = p[j], p[i]
            k = pow2 // 2
            while True:
                if k > j:
                    break
                j -= k
                k //= 2
            j += k
        l = 2
        m = len(r) * forward
        for _ in range(logpow2):
            for j in range(l // 2):
                for k in range(j, pow2, l):
                    a = p[k]
                    b = r[j * m // l] * p[k + l // 2]
                    b %= mod
                    p[k] = (a + b) % mod
                    p[k + l // 2] = (a - b) % mod
            l *= 2
        if forward == -1:
            for i in range(pow2):
                p[i] *= invpow2
                p[i] %= mod

    @staticmethod
    def get_answer(c, v, n, s, k):
        mod = PowerSumNTT.mod
        pow2, logpow2, invpow2 = PowerSumNTT.get_pow2(2 * k)
        prim_root = PowerSumNTT.power(3, (mod - 1) // pow2)
        r = [1] * pow2
        for i in range(1, pow2):
            r[i] = r[i - 1] * prim_root
            r[i] %= mod
        factorial = [1] * pow2
        factorial_inv = [1] * pow2
        for i in range(1, pow2):
            factorial[i] = factorial[i - 1] * i
            factorial[i] %= mod
            factorial_inv[i] = PowerSumNTT.power(factorial[i], mod - 2)
        vpowers = [[1] * pow2 for _ in range(n + 1)]
        for i in range(1, n + 1):
            for l in range(1, pow2):
                vpowers[i][l] = vpowers[i][l - 1] * v[i]
                vpowers[i][l] %= mod
        dp = [[0] * (k + 1) for _ in range(s + 1)]
        dp[0][0] = 1
        p1 = [0] * pow2
        p2 = [0] * pow2
        for i in range(1, n + 1):
            for j in range(s, c[i] - 1, -1):
                for l in range(pow2):
                    if l <= k:
                        p1[l] = dp[j - c[i]][l] * factorial_inv[l] % mod
                        p2[l] = factorial_inv[l] * vpowers[i][l] % mod
                    else:
                        p1[l] = 0
                        p2[l] = 0
                PowerSumNTT.NTT(p1, r, pow2, logpow2, invpow2, 1)
                PowerSumNTT.NTT(p2, r, pow2, logpow2, invpow2, 1)
                for l in range(pow2):
                    p1[l] *= p2[l]
                    p1[l] %= mod
                PowerSumNTT.NTT(p1, r, pow2, logpow2, invpow2, -1)
                for l in range(k + 1):
                    dp[j][l] += p1[l] * factorial[l]
                    dp[j][l] %= mod
        answer = 0
        for j in range(s + 1):
            answer += dp[j][k]
            answer %= mod
        return answer

    @staticmethod
    def solve_from_stdin():
        import sys
        n, s, k = list(map(int, sys.stdin.readline().strip().split()))
        c = [0] * (n + 1)
        v = [0] * (n + 1)
        for i in range(n):
            c[i + 1], v[i + 1] = list(map(int, sys.stdin.readline().strip().split()))
        ans = PowerSumNTT.get_answer(c, v, n, s, k)
        print(ans)
        return ans





# =============================================================================
# ── CONSOLIDATED FROM temp.py (ADDITIONAL & COMMENTED PROBLEMS) ──────────────
# =============================================================================


class SnippetExamples:
    """Small one-off snippet demonstrations (Problems 0-6 from temp.py)."""

    def left_rotate(arr, d):
        """Left-rotate array by d steps. Example: left_rotate([1,2,3,4,5], 4) -> [5,1,2,3,4]"""
        arr = list(arr)
        d %= len(arr)
        return arr[d:] + arr[:d]

    def invert_dict(my_dict):
        """Group dict keys by value. Example: invert_dict({"a":"x","b":"x"}) -> {"x":["a","b"]}"""
        inv = {}
        for key, val in my_dict.items():
            inv.setdefault(val, []).append(key)
        return inv

    def reduce_product(lst):
        """Product of all elements. Example: reduce_product([1,2,3,4]) -> 24"""
        from functools import reduce
        return reduce(lambda x, y: x * y, lst)

    def diagonal_difference(a):
        """Absolute difference of matrix diagonals."""
        n = len(a)
        return abs(sum(a[i][i] - a[i][n-i-1] for i in range(n)))

    def palindrome(word):
        """Check palindrome. Example: palindrome("abcba") -> True"""
        return all(word[i] == word[-i-1] for i in range(len(word)//2))

    def array_manipulation(n, queries):
        """
        HackerRank array manipulation (prefix sum trick).
        Example: array_manipulation(5, [[1,2,100],[2,5,100],[3,4,100]]) -> 200
        """
        arr = [0] * (n + 2)
        for a, b, k in queries:
            arr[a] += k
            arr[b+1] -= k
        result = acc = 0
        for x in arr:
            acc += x
            result = max(result, acc)
        return result


class SubsetProblems:
    """Subset sum and k-partition problems."""

    def isSubsetSum_recursive(s, n, total):
        """
        (naive) Recursive. Time O(2^n).
        Example: SubsetProblems.isSubsetSum_recursive([3,34,4,12,5,2], 6, 9) -> True
        """
        if total == 0:
            return True
        if n == 0:
            return False
        if s[n-1] > total:
            return SubsetProblems.isSubsetSum_recursive(s, n-1, total)
        return (SubsetProblems.isSubsetSum_recursive(s, n-1, total) or
                SubsetProblems.isSubsetSum_recursive(s, n-1, total - s[n-1]))

    def isSubsetSum_dp(s, n, total):
        """
        (best) DP table. Time O(n*total).
        Example: SubsetProblems.isSubsetSum_dp([3,34,4,12,5,2], 6, 9) -> True
        """
        dp = [[False] * (total + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = True
        for i in range(1, n + 1):
            for j in range(1, total + 1):
                dp[i][j] = dp[i-1][j] or (j >= s[i-1] and dp[i-1][j - s[i-1]])
        return dp[n][total]

    def subset(array, num):
        """Find all subsets summing to num. Example: SubsetProblems.subset([3,4,2,1], 5) -> [(4,1),(3,2)]"""
        result = []
        def find(arr, rem, path=()):
            if not arr:
                return
            if arr[0] == rem:
                result.append(path + (arr[0],))
            else:
                find(arr[1:], rem - arr[0], path + (arr[0],))
                find(arr[1:], rem, path)
        find(array, num)
        return result

    def k_partition(S, k):
        """
        Partition S into k subsets with equal sum (backtracking).
        Example: SubsetProblems.k_partition([7,3,5,12,2,1,5,3,8,4,6,4], 5) -> list of 5 partitions
        """
        n, total = len(S), sum(S)
        if n < k or total % k:
            return None
        A = [None] * n
        sl = [total // k] * k

        def bt(idx):
            if all(x == 0 for x in sl):
                return True
            if idx < 0:
                return False
            for i in range(k):
                if sl[i] >= S[idx]:
                    A[idx] = i + 1
                    sl[i] -= S[idx]
                    if bt(idx - 1):
                        return True
                    sl[i] += S[idx]
            return False

        if bt(n - 1):
            return [[S[j] for j in range(n) if A[j] == i+1] for i in range(k)]
        return None


class DifferenceArray:
    """Range update via difference arrays."""

    def initialize(A):
        """Create diff array D from A. Example: DifferenceArray.initialize([10,5,20,40])"""
        n = len(A)
        D = [0] * (n + 1)
        D[0] = A[0]
        for i in range(1, n):
            D[i] = A[i] - A[i-1]
        return D

    def update(D, l, r, x):
        """Add x to all elements in [l, r]. Example: DifferenceArray.update(D, 0, 1, 10)"""
        D[l] += x
        D[r+1] -= x

    def reconstruct(A, D):
        """Reconstruct updated array. Example: DifferenceArray.reconstruct(A, D)"""
        res = list(A)
        for i in range(len(A)):
            res[i] = D[i] if i == 0 else D[i] + res[i-1]
        return res


class StockProfit:
    """Max profit with k stock transactions."""

    def max_profit_kn(prices, k):
        """
        (naive) O(nk) time and space.
        Example: StockProfit.max_profit_kn([50,25,12,4,3,10,1,100], 2) -> 97
        """
        if not prices:
            return 0
        profit = [[0] * len(prices) for _ in range(k + 1)]
        for t in range(1, k + 1):
            mtf = float("-inf")
            for d in range(1, len(prices)):
                mtf = max(mtf, profit[t-1][d-1] - prices[d-1])
                profit[t][d] = max(profit[t][d-1], mtf + prices[d])
        return profit[k][-1]

    def max_profit_kn_optimized(prices, k):
        """
        (best) O(n) space via two rolling arrays.
        Example: StockProfit.max_profit_kn_optimized([50,25,12,4,3,10,1,100], 2) -> 97
        """
        if not prices:
            return 0
        even, odd = [0] * len(prices), [0] * len(prices)
        for t in range(1, k + 1):
            mtf = float("-inf")
            cur, prev = (odd, even) if t % 2 else (even, odd)
            for d in range(1, len(prices)):
                mtf = max(mtf, prev[d-1] - prices[d-1])
                cur[d] = max(cur[d-1], mtf + prices[d])
        return even[-1] if k % 2 == 0 else odd[-1]


class RiverSizes:
    """Connected river (1-cell) sizes in a binary matrix."""

    def river_sizes(matrix):
        """
        Return list of river sizes.
        Example: RiverSizes.river_sizes([[1,0,0],[1,1,0],[0,0,1]]) -> [3, 1]
        """
        sizes, visited = [], [[False]*len(r) for r in matrix]
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if not visited[i][j]:
                    RiverSizes._traverse(i, j, matrix, visited, sizes)
        return sizes

    def _traverse(i, j, matrix, visited, sizes):
        size, stack = 0, [[i, j]]
        while stack:
            r, c = stack.pop()
            if visited[r][c]:
                continue
            visited[r][c] = True
            if matrix[r][c] == 0:
                continue
            size += 1
            for nr, nc in RiverSizes._neighbors(r, c, matrix, visited):
                stack.append([nr, nc])
        if size > 0:
            sizes.append(size)

    def _neighbors(i, j, matrix, visited):
        res = []
        if i > 0 and not visited[i-1][j]: res.append([i-1, j])
        if i < len(matrix)-1 and not visited[i+1][j]: res.append([i+1, j])
        if j > 0 and not visited[i][j-1]: res.append([i, j-1])
        if j < len(matrix[0])-1 and not visited[i][j+1]: res.append([i, j+1])
        return res


class ArrayUtils:
    """Miscellaneous array, string, and numeric utilities."""

    def largest_range(array):
        """
        Longest consecutive integer range.
        Example: ArrayUtils.largest_range([1,11,3,0,15,5,2,4,10,7,12,6]) -> [0, 7]
        """
        best, longest, nums = [], 0, {n: True for n in array}
        for num in array:
            if not nums[num]:
                continue
            nums[num] = False
            length, left, right = 1, num-1, num+1
            while left in nums: nums[left] = False; length += 1; left -= 1
            while right in nums: nums[right] = False; length += 1; right += 1
            if length > longest:
                longest, best = length, [left+1, right-1]
        return best

    def birthday(s, d, m):
        """
        Count subarrays of length m summing to d.
        Example: ArrayUtils.birthday([1,2,1,3,2], 3, 2) -> 2
        """
        return sum(1 for x in range(len(s)) if sum(s[x:x+m]) == d)

    def bisection(f, a, b, N):
        """
        Bisection root-finding. Example: ArrayUtils.bisection(lambda x: x**2-x-1, 1, 2, 25) -> ~1.618
        """
        if f(a) * f(b) >= 0:
            return None
        a_n, b_n = a, b
        for _ in range(N):
            m = (a_n + b_n) / 2
            fm = f(m)
            if fm == 0:
                return m
            elif f(a_n) * fm < 0:
                b_n = m
            elif f(b_n) * fm < 0:
                a_n = m
            else:
                return None
        return (a_n + b_n) / 2

    def h_index(lst):
        """Academic h-index. Example: ArrayUtils.h_index([4,1,0,2,3]) -> 2"""
        result = 0
        for i, c in enumerate(sorted(lst, reverse=True)):
            if c > i:
                result = i + 1
            else:
                break
        return result

    def sliding_window_median(lst, k):
        """
        Sliding window median.
        Example: ArrayUtils.sliding_window_median([-1,5,13,8,2,3,3,1], 5) -> list of medians
        """
        from bisect import insort
        window = sorted(lst[:k])
        results = [(window[k//2] + window[~(k//2)]) / 2.0]
        for remove, add in zip(lst, lst[k:]):
            window.remove(remove)
            insort(window, add)
            results.append((window[k//2] + window[~(k//2)]) / 2.0)
        return results

    def xor_decipher(s):
        """
        Brute-force XOR decipher from hex.
        Example: ArrayUtils.xor_decipher("7a575e5e5d12455d405e56...") -> list of (key, text) candidates
        """
        b = bytearray.fromhex(s)
        results = []
        for char in range(256):
            try:
                results.append((char, bytes([byte ^ char for byte in b]).decode()))
            except Exception:
                pass
        return results

    def find_missing_nums(lst):
        """
        Find missing numbers from 1..1,000,000.
        Example: ArrayUtils.find_missing_nums(range(1, 999001)) -> [999001..1000000]
        """
        s = set(lst)
        return [i for i in range(1, 1_000_001) if i not in s]

    def longest_contiguous_history(user1, user2):
        """
        Longest common subarray of page visits.
        Example: ArrayUtils.longest_contiguous_history(["/a","/b","/c"],["/x","/b","/c"]) -> ["/b","/c"]
        """
        longest = []
        for i in range(len(user1)):
            for j in range(i+1, len(user1)+1):
                sub = user1[i:j]
                for k in range(len(user2) - len(sub) + 1):
                    if sub == user2[k:k+len(sub)] and len(sub) > len(longest):
                        longest = sub
        return longest

    def island_perimeter(board):
        """
        Island perimeter in 0/1 matrix.
        Example: ArrayUtils.island_perimeter([[0,1,1,0],[1,1,1,0],[0,1,1,0],[0,0,1,0]]) -> 14
        """
        def nb(r, c):
            n = 0
            if r > 0: n += board[r-1][c] == 1
            if r < len(board)-1: n += board[r+1][c] == 1
            if c > 0: n += board[r][c-1] == 1
            if c < len(board[0])-1: n += board[r][c+1] == 1
            return n
        return sum(4 - nb(r, c) for r, row in enumerate(board)
                   for c, val in enumerate(row) if val == 1)

    def group_anagrams(words):
        """
        Group words by anagram signature.
        Example: ArrayUtils.group_anagrams(["eat","ate","apt","pat","tea","now"])
        """
        from collections import defaultdict
        groups = defaultdict(list)
        for w in words:
            groups["".join(sorted(w))].append(w)
        return list(groups.values())

    def compress_array_continuous(arr):
        """
        (naive) Compress consecutive runs only.
        Example: ArrayUtils.compress_array_continuous([1,1,1,4,4,3]) -> [1,3,4,2,3,1]
        """
        if not arr:
            return []
        out, count, ch = [], 1, arr[0]
        for i in range(1, len(arr)):
            if arr[i] == ch:
                count += 1
            else:
                out.extend([ch, count])
                ch, count = arr[i], 1
        out.extend([ch, count])
        return out

    def compress_array_any(arr):
        """
        (best) Compress any elements preserving first-seen order.
        Example: ArrayUtils.compress_array_any([1,1,1,4,4,3,1,3]) -> [1,4,4,2,3,2]
        """
        from collections import OrderedDict
        d = OrderedDict.fromkeys(arr, 0)
        for x in arr:
            d[x] += 1
        out = []
        for k, v in d.items():
            out.extend([k, v])
        return out


class GraphProblems:
    """Graph connectivity and scheduling."""

    def _find(parents, i):
        if parents[i] != i:
            parents[i] = GraphProblems._find(parents, parents[i])
        return parents[i]

    def components_in_graph(gb):
        """
        (min, max) component sizes in edge list.
        Example: GraphProblems.components_in_graph([[1,2],[3,4],[1,4]]) -> (2, 4)
        """
        parents = list(range(len(gb) * 2 + 1))
        for a, b in gb:
            p1, p2 = GraphProblems._find(parents, a), GraphProblems._find(parents, b)
            parents[p1] = parents[p2] = parents[a] = parents[b] = min(p1, p2)
        from collections import Counter
        cnt = Counter(GraphProblems._find(parents, p) for p in parents)
        counts = [c for c in cnt.values() if c > 1]
        return min(counts), max(counts)

    def minimum_average(cust):
        """
        Min average wait time (heap scheduling).
        Example: GraphProblems.minimum_average([[0,3],[1,9],[2,5]]) -> 5
        """
        from heapq import heapify, heappop, heappush
        cust = list(cust)
        n = len(cust)
        if not n:
            return 0
        heapify(cust)
        tl, done, orders = 0, 0, []
        while orders or cust:
            while ((not cust) or done < cust[0][0]) and orders:
                dw, dt = heappop(orders)
                done = dw + max(done, dt)
                tl += done - dt
            if cust:
                heappush(orders, heappop(cust)[::-1])
        return tl // n


class UnionFind:
    """Disjoint Set Forest with rank and path compression."""

    class UFNode:
        def __init__(self, data):
            self.data = data
            self.parent = self
            self.rank = 0
            self.size = 1

    def make_set(data):
        """Example: node = UnionFind.make_set(1)"""
        return UnionFind.UFNode(data)

    def find(node):
        """Find with path compression."""
        if node != node.parent:
            node.parent = UnionFind.find(node.parent)
        return node.parent

    def union(node_a, node_b):
        """Union by rank. Example: UnionFind.union(a, b)"""
        ra, rb = UnionFind.find(node_a), UnionFind.find(node_b)
        if ra == rb:
            return
        if ra.rank > rb.rank:
            rb.parent = ra; ra.size += rb.size
        else:
            ra.parent = rb; rb.size += ra.size
            if ra.rank == rb.rank:
                rb.rank += 1


class BSTShowcase:
    """BST construction, validation, and largest-BST-subtree problems."""

    class BSTNode:
        def __init__(self, data):
            self.data = data
            self.left_child = None
            self.right_child = None

    class BST:
        """
        Full BST. Example:
            t = BSTShowcase.BST()
            for v in [10,15,6,4,9]: t.insert(v)
            t.inorder()
        """
        def __init__(self): self.root = None

        def insert(self, data):
            if not self.root:
                self.root = BSTShowcase.BSTNode(data)
            else:
                self._ins(self.root, data)

        def _ins(self, n, d):
            if d < n.data:
                if n.left_child: self._ins(n.left_child, d)
                else: n.left_child = BSTShowcase.BSTNode(d)
            elif d > n.data:
                if n.right_child: self._ins(n.right_child, d)
                else: n.right_child = BSTShowcase.BSTNode(d)

        def search(self, data): return self._srch(self.root, data)

        def _srch(self, n, d):
            if not n: return False
            if n.data == d: return True
            return self._srch(n.right_child if d > n.data else n.left_child, d)

        def inorder(self): self._io(self.root); print("End")

        def _io(self, n):
            if not n: return
            self._io(n.left_child); print(n.data, "->", end=" "); self._io(n.right_child)

    def is_bst(root):
        """Check BST validity. Example: BSTShowcase.is_bst(root) -> True/False"""
        def _chk(n, lo, hi):
            if not n: return True
            if n.data <= lo or n.data >= hi: return False
            return _chk(n.left_child, lo, n.data) and _chk(n.right_child, n.data, hi)
        return _chk(root, float("-inf"), float("inf"))

    def size(root):
        """Count nodes. Example: BSTShowcase.size(root) -> int"""
        if not root: return 0
        return 1 + BSTShowcase.size(root.left_child) + BSTShowcase.size(root.right_child)

    def largest_bst_subtree_naive(root):
        """
        (naive) Checks is_bst on every node. O(n^2).
        Example: BSTShowcase.largest_bst_subtree_naive(root)
        """
        if not root: return None
        if BSTShowcase.is_bst(root): return root
        return max(
            BSTShowcase.largest_bst_subtree_naive(root.left_child),
            BSTShowcase.largest_bst_subtree_naive(root.right_child),
            key=lambda r: BSTShowcase.size(r) if r else 0
        )

    def largest_bst_subtree_optimized(root):
        """
        (best) Single-pass O(n).
        Example: BSTShowcase.largest_bst_subtree_optimized(root)
        """
        best = [0, None]
        def _h(n):
            if not n: return (0, float("inf"), float("-inf"))
            l, r = _h(n.left_child), _h(n.right_child)
            if n.data > l[2] and n.data < r[1]:
                sz = l[0] + r[0] + 1
                if sz > best[0]: best[0] = sz; best[1] = n
                return (sz, min(n.data, l[1]), max(n.data, r[2]))
            return (0, float("-inf"), float("inf"))
        _h(root)
        return best[1]


class CameraCoverSolution:
    """
    Minimum cameras to cover all binary tree nodes (LeetCode #968).
    Example:
        root = CameraCoverSolution(0)
        root.left = CameraCoverSolution(0)
        root.left.left = CameraCoverSolution(0)
        print(root.min_camera_cover())  # -> 1
    """
    def __init__(self, k):
        self.key = k; self.left = None; self.right = None; self.ans = 0

    def min_camera_cover(self):
        def dfs(n):
            if not n: return 0
            v = dfs(n.left) + dfs(n.right)
            if v == 0: return 3
            if v < 3: return 0
            self.ans += 1; return 1
        return self.ans + 1 if dfs(self) > 2 else self.ans


class SynonymQueries:
    """Sentence equivalence via synonym lookup."""

    def solve_naive(synonym_words, queries):
        """
        (naive) defaultdict — no transitive synonyms.
        Example: SynonymQueries.solve_naive([("big","large")],[("He is big.","He is large.")]) -> [True]
        """
        from collections import defaultdict
        syn = defaultdict(set)
        for w1, w2 in synonym_words:
            syn[w1].add(w2)
        out = []
        for q1, q2 in queries:
            q1, q2 = q1.split(), q2.split()
            if len(q1) != len(q2):
                out.append(False); continue
            out.append(all(
                w1 == w2 or (w1 in syn and w2 in syn[w1]) or (w2 in syn and w1 in syn[w2])
                for w1, w2 in zip(q1, q2)
            ))
        return out

    def solve_disjoint_set(synonym_words, queries):
        """
        (best) DisjointSet path compression — handles transitivity.
        Example: SynonymQueries.solve_disjoint_set([("big","large"),("large","huge")],[("big","huge")]) -> [True]
        """
        class _DS:
            def __init__(self): self.p = {}
            def root(self, w):
                if w not in self.p: self.p[w] = w
                path = []
                while self.p[w] != w: path.append(w); w = self.p[w]
                for x in path: self.p[x] = w
                return w
            def add(self, a, b):
                ra, rb = self.root(a), self.root(b)
                if ra > rb: ra, rb = rb, ra
                self.p[rb] = ra
            def same(self, a, b): return self.root(a) == self.root(b)

        ds = _DS()
        for w1, w2 in synonym_words:
            ds.add(w1, w2)
        out = []
        for q1, q2 in queries:
            q1, q2 = q1.split(), q2.split()
            if len(q1) != len(q2):
                out.append(False); continue
            out.append(all(w1 == w2 or ds.same(w1, w2) for w1, w2 in zip(q1, q2)))
        return out
