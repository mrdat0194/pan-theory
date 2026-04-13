#!/bin/python3
import os
import itertools
from collections import defaultdict, Counter
from functools import cmp_to_key
import heapq
from typing import Set, List

from my_functions.timer import print_param, timer

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

    @print_param("twoString.txt", BASE_DIR)
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

    @print_param("SherlockAna.txt", BASE_DIR)
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

    @print_param("valleycount.txt", BASE_DIR)
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


    @print_param("repeatedstrings.txt", BASE_DIR)
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


    @print_param("compareTriplets.txt", BASE_DIR)
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

from bisect import bisect_left,bisect_right
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
import heapq
from copy import copy

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







