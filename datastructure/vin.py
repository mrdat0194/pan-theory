# Question 1. Given an array of integer numbers, which are already sorted. E.g., A = [1,2,3,3,3,4,4,5,5,6]
# •	Find the mode of the array
# •	Provide the time complexity and space complexity of the array, and your reasoning
# •	Note: write your own function using the basic data structure of your language, please avoid the provided available functions from external lib


# Question 2. Given the stock value of VIC in n continuous days as an array, e.g., A = [p1, p2, p3, ..., pn]
# Please find a single buying date and a single selling date so that the the buyer could get the best profit.
# Knowing that you have to buy before selling anything.

def ModedSortedArray(arr):
    length = len(arr)
    count = 1
    tempMode = arr[0]
    tempCount = 1
    modes = 0
    for i in range(1,length):
        if tempMode == arr[i]:
            tempCount += 1
            if tempCount > count:
                modes = tempMode
                count = tempCount
        else:
            tempMode = arr[i]
            tempCount = 1
    return modes

def ModedArray(arr):
    count = 1
    tempCount = 0
    modes = arr[0]
    for i in range(len(arr) - 1):
        tempMode = arr[i]
        print(tempMode)
        for j in range(1, len(arr)):
            if tempMode == arr[j]:
                tempCount += 1
        if tempCount > count:
            modes = tempMode
            count = tempCount
    return modes

def maxProfitWithKTransactions(prices, k):
    # Time: O(kn) | Space: O(n)
    prev_trans = [0 for _ in prices]
    curr_trans = []
    for t in range(k):
        curr_trans = [0]
        max_diff = float('-inf')

        for i in range(1, len(prices)):
            max_diff = max(max_diff, prev_trans[i - 1] - prices[i - 1])
            curr_trans.append(max(curr_trans[i - 1], prices[i] + max_diff))
        prev_trans = curr_trans[:]
        print(curr_trans)
    return curr_trans[-1]


def get_diff(s1, s1_sum, s2, s2_sum, score):
    """
        # Given an array of positive integers, divide the array into two subsets such
        # that the difference between the sum of the subsets is as small as possible.
        #
        # For example, given [5, 10, 15, 20, 25], return the sets {10, 25} and {5, 15, 20},
        # which has a difference of 5, which is the smallest possible difference.

    :param s1:
    :param s1_sum:
    :param s2:
    :param s2_sum:
    :param score:
    :return:
    """
    min_diff, min_cand = score, None
    for i, num in enumerate(s1):

        new_s1_sum, new_s2_sum = s1_sum - num, s2_sum + num
        new_score = abs(new_s1_sum - new_s2_sum)
        if new_score < min_diff:
            min_diff = new_score
            min_cand = (s1[:i] + s1[i + 1:], new_s1_sum,
                        s2 + [num], new_s2_sum)
    if not min_cand:
        return (set(s1), set(s2))

    return get_diff(min_cand[0], min_cand[1], min_cand[2], min_cand[3], min_diff)


def divide_numbers(nums):
    sum_nums = sum(nums)
    best_sets = get_diff(nums.copy(), sum_nums, [], 0, sum_nums)

    return best_sets
# Tests
assert divide_numbers([5, 10, 15, 20, 25]) == ({5, 15, 20}, {10, 25})
assert divide_numbers([5, 10, 15, 20]) == ({10, 15}, {20, 5})


#dynamic programming FIB

def fib_dn(N, memo={0:1,1:1}):
    if N not in memo:
        memo[N] = fib_dn(N-1,memo) + fib_dn(N-2,memo)
    return memo[N]

def is_prime(x):
    '''

    the original prime functions that could be improved.
        prime_or_not = is_prime(991)
        print(prime_or_not)
    :param x:
    :return:
    '''
    for i in range(2, int(x**0.5) + 1):
        if x%i == 0:
            return False
        else:
            print(x)
            return True

# Python program to display all the prime numbers within an interval
def prime_interval(x):
    """
        # Python program to print
        # print all primes in a range
        # using concept of Segmented Sieve
    :param x:
    :return:
    """
    lower = 900
    upper = 1000
    print("Prime numbers between", lower, "and", upper, "are:")

    for num in range(lower, upper + 1):
        # all prime numbers are greater than 1
        if num > 1:
            for i in range(2, num):
                if (num % i) == 0:
                    break
            else:
                print(num)


from math import floor, sqrt  # noqa: E402


def simpleSieve(limit, primes):
    """

        # This functions finds
        # all primes smaller than limit
        # using simple sieve of eratosthenes.
        # It stores found
        # primes in list prime[]
    :param limit:
    :param primes:
    :return:
    """
    mark = [False]*(limit+1)

    for i in range(2, limit+1):
        if not mark[i]:

            # If not marked yet,
            # then its a prime
            primes.append(i)
            for j in range(i, limit+1, i):
                mark[j] = True


# Finds all prime numbers
# in given range using
# segmented sieve
def primesInRange(low, high):
    """
        # Driver code
        # low = 10
        # high = 100
        # primesInRange(low, high)
    :param low:
    :param high:
    :return:
    """

    # Comput all primes smaller
    # or equal to
    # square root of high
    # using simple sieve
    limit = floor(sqrt(high)) + 1
    primes = list()
    simpleSieve(limit, primes)

    # Count of elements in given range
    n = high - low + 1

    # Declaring boolean only for
    # [low, high]
    mark = [False]*(n+1)

    # Use the found primes by
    # simpleSieve() to find
    # primes in given range
    for i in range(len(primes)):

        # Find the minimum number
        # in [low..high] that is
        # a multiple of prime[i]
        # (divisible by prime[i])
        loLim = floor(low/primes[i]) * primes[i]
        if loLim < low:
            loLim += primes[i]
        if loLim == primes[i]:
            loLim += primes[i]

        # Mark multiples of primes[i]
        # in [low..high]:
        # We are marking j - low for j,
        # i.e. each number
        # in range [low, high] is mapped
        # to [0, high-low]
        # so if range is [50, 100]
        # marking 50 corresponds
        # to marking 0, marking 51
        # corresponds to 1 and
        # so on. In this way we need
        # to allocate space
        # only for range
        for j in range(loLim, high+1, primes[i]):
            mark[j-low] = True

    # Numbers which are not marked
    # in range, are prime
    for i in range(low, high+1):
        if not mark[i-low]:
            print(i, end=" ")


def my_fib(x, memo=dict()):
    if memo.get(x):
        return memo[x]
    if x == 1 or x == 2:
        result = 1
    else:
        result = my_fib(x - 1, memo) + my_fib(x -2, memo)
    memo[x] = result
    return result

def fibo(N):
    a = b = 1
    for _ in range(2,N): a,b = b,a+b
    return b

def simpleFibo(N,a=0,b=1):
    if N < 3: return a+b
    return simpleFibo(N-1,b,a+b)

def dynaFibo(N,memo={1:1,2:1}):
    '''Dynamic Programming'''
    if N not in memo:
        memo[N] = dynaFibo(N-1,memo) + dynaFibo(N-2,memo)
    return memo[N]

def dynaFibo2(N,memo=None):
    '''Dynamic Programming'''
    if not memo:    memo = [0,1,1]+[0]*N
    if not memo[N]: memo[N] = dynaFibo2(N-1,memo) + dynaFibo2(N-2,memo)
    return memo[N]

cache = [0, 1]   # Initialize with the first two terms of Fibonacci series.
def fibonacci(n):
    """
    Returns the n-th number in the Fibonacci sequence.
    bottom up
    Parameters
    ----------
    n: int
       The n-th number in the Fibonacci sequence.
    """
    for i in range(2, n):
        cache.append(cache[i-1] + cache[i-2])
    return cache[-1]

def binFibo(N):
    a,b   = 0,1
    f0,f1 = 1,1
    r,s   = (1,1) if N&1 else (0,1)
    N //=2
    while N > 0:
        a,b   = f0*a+f1*b, f0*b+f1*(a+b)
        f0,f1 = b-a,a
        if N&1: r,s = f0*r+f1*s, f0*s+f1*(r+s)
        N //= 2
    return r

def test_fib():
    from timeit import timeit
    count = 100

    N = 990

    t= timeit(lambda:my_fib(N,dict()), number=count) # providing dict() to avoid reuse between repetitions
    print("my_fib(N)",t)

    t= timeit(lambda:fibo(N), number=count)
    print("fibo(N)",t)

    t= timeit(lambda:simpleFibo(N), number=count)
    print("simpleFibo(N)",t)

    t= timeit(lambda:dynaFibo(N,{1:1,2:1}), number=count) # providing dict() to avoid reuse between repetitions
    print("dynaFibo(N)",t)

    t= timeit(lambda:dynaFibo2(N), number=count)
    print("dynaFibo2(N)",t)

    t= timeit(lambda:binFibo(N), number=count)
    print("binFibo(N)",t)



def rectangles(rec1, rec2):
    """
        #intersect rectangle
        # Given two rectangles on a 2D graph, return the area of their intersection. If the rectangles don't intersect, return 0.
        #
        # For example, given the following rectangles:
        #
        # {
        #     "top_left": (1, 4),
        #     "dimensions": (3, 3) # width, height
        # }
        # and
        #
        # {
        #     "top_left": (0, 5),
        #     "dimensions": (4, 3) # width, height
        # }
        # return 6.
    :param rec1:
    :param rec2:
    :return:
    """
    left_x = max(rec1["top_left"][0], rec2["top_left"][0])
    right_x = min(rec1["top_left"][0] + rec1["dimensions"][0], rec2["top_left"][0] + rec2["dimensions"][0])

    top_y = min(rec1["top_left"][1], rec2["top_left"][1])
    bottom_y = max(rec1["top_left"][1] - rec1["dimensions"][1], rec2["top_left"][1] - rec2["dimensions"][1])

    if left_x > right_x or bottom_y > top_y:
        return 0

    return (right_x - left_x) * (top_y - bottom_y)

### others

def sum(array):
    """
        array = list(map(int, input().split()))
        print(sum(array))
    :param array:
    :return:
    """
    return array[0] + array[1]


def _lis(arr, n):

    # to allow the access of global variable
    global maximum

    # Base Case
    if n == 1:
        return 1

    # maxEndingHere is the length of LIS ending with arr[n-1]
    maxEndingHere = 1

    """Recursively get all LIS ending with arr[0], arr[1]..arr[n-2]
       IF arr[n-1] is maller than arr[n-1], and max ending with
       arr[n-1] needs to be updated, then update it"""
    for i in range(1, n):
        res = _lis(arr, i)
        if arr[i-1] < arr[n-1] and res+1 > maxEndingHere:
            maxEndingHere = res + 1

    # Compare maxEndingHere with overall maximum. And
    # update the overall maximum if needed
    maximum = max(maximum, maxEndingHere)

    return maxEndingHere


def continuous_incr(arr,n):
    """
    n = int(input())
    array = list(map(int, input().split()))

    print(continuous_incr(array,n))
6
1 2 5 4 6 2
    :param arr:
    :param n:
    :param maximum:
    :return:
    """
    global maximum

    maximum = 1

    # The function _lis() stores its result in maximum
    _lis(arr, n)

    return maximum

def lengthOfLIS(nums,n):
    """
    n = int(input())
    array = list(map(int, input().split()))

    print(lengthOfLIS(array,n))

6
1 2 5 4 6 2

    :type nums: List[int]
    :rtype: int
    """
    LIS = []
    def insert(target):
        left, right = 0, len(LIS) - 1
        # Find the first index "left" which satisfies LIS[left] >= target
        while left <= right:
            mid = left + (right - left) // 2
            if LIS[mid] >= target:
                right = mid - 1
            else:
                left = mid + 1
        # If not found, append the target.
        if left == len(LIS):
            LIS.append(target);
        else:
            LIS[left] = target

    for num in nums:
        insert(num)

    return len(LIS)


def isPresent(arr, low, high, value):

    while (low <= high):

        mid = (low + high) // 2

        # value found
        if (arr[mid] == value):
            return True

        elif (arr[mid] > value) :
            high = mid - 1
        else:
            low = mid + 1

    # value not found
    return False

# function to count all pairs
# from both the sorted arrays
# whose sum is equal to a given
# value
def countPairs(arr1, arr2, m, n, x):
    """
    count = 0
    m = len(arr1)
    n = len(arr2)

    arr1 = sorted(arr1)
    arr2 = sorted(arr2)

    for x in arr3:
        count += countPairs(arr1, arr2,m ,n, x)
    :param arr1:
    :param arr2:
    :param m:
    :param n:
    :param x:
    :return:
    """
    count = 0
    for i in range(m):
        # for each arr1[i]
        value = x - arr1[i]

        # check if the 'value'
        # is present in 'arr2[]'
        if (isPresent(arr2, 0, n - 1, value)):
            count += 1

    # required count of pairs
    return count

from collections import Counter  # noqa: E402

def countPairs3(arr1, arr2, arr3):
    """
    n = int(input())
    arr1 = list(map(int, input().split()))
    arr2 = list(map(int, input().split()))
    arr3 = list(map(int, input().split()))

    # arr1 = [1, 5, 3, 7]
    # arr2 = [2, 3, 5, 8]
    # arr3 = [10, 3]

    # 3
    # -1 1 1
    # -1 2 3
    # 2 3 -2

    m = len(arr1)
    n = len(arr2)
    lookup = dict()

    arr1 = sorted(arr1)
    arr2 = sorted(arr2)

    # count = countPairs(arr1, arr2, arr3)
    #
    # print(count)


    # from itertools import permutations
    #
    # permut = itertools.permutations(arr1, len(arr2))
    #
    # unique_combinations = []
    #
    # for comb in permut:
    #     zipped = zip(comb, arr2)
    #     unique_combinations.append(list(zipped))
    #
    # # printing unique_combination list
    # print(unique_combinations)
    # https://stackoverflow.com/questions/54927383/itertools-product-function-with-sum

    import itertools

    a = [arr1, arr2, arr3]
    permute = itertools.product(*a)
    counts = 0
    for i in permute:
        if (i[0] + i[1] == i[2]):
            counts += 1

    print(counts)

    :param arr1:
    :param arr2:
    :param arr3:
    :return:
    """
    count = 0
    m = Counter(arr3)
    n = Counter(arr2)
    counts = Counter(arr1)

    for i in m:
        for t in n:
            complement = i - t
            if complement in counts.keys():
                count += counts.get(complement) * m.get(i) * n.get(t)
    return count


# nelder-mead for multimodal function optimization

# from scipy.optimize import minimize
# from numpy.random import rand
# from numpy import exp
# from numpy import sqrt
# from numpy import cos
# from numpy import e
# from numpy import pi


# # objective function
# def objective(v):
#     x, y = v
#     return -20.0 * exp(-0.2 * sqrt(0.5 * (x ** 2 + y ** 2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20
#
# # define range for input
# r_min, r_max = -5.0, 5.0
# # define the starting point as a random sample from the domain
# pt = r_min + rand(2) * (r_max - r_min)
# # perform the search
# result = minimize(objective, pt, method='nelder-mead')
# # summarize the result
# print('Status : %s' % result['message'])
# print('Total Evaluations: %d' % result['nfev'])
# # evaluate solution
# solution = result['x']
# evaluation = objective(solution)
# print('Solution: f(%s) = %.5f' % (solution, evaluation))

# # Power Sum Problem: watch fft/nnt
    # # 3 2 2 : means 3 ingredients - 2 budget cost  - fix power x^2
    # # 1 2 : cost - x
    # # 2 3 : cost - x
    # # 1 4 : cost - x
    # from sys import stdin
    #
    # mod = 998244353
    #
    #
    # def power(n, k):
    #     if k == 0:
    #         return 1
    #     x = 1
    #     while k > 1:
    #         if k % 2 == 0:
    #             n = n * n % mod
    #         else:
    #             x = n * x % mod
    #             n = n * n % mod
    #         k //= 2
    #     return n * x % mod
    #
    #
    # def get_pow2(k):
    #     pow2 = 1
    #     logpow2 = 0
    #     while pow2 <= k:
    #         pow2 *= 2
    #         logpow2 += 1
    #     invpow2 = power(pow2, mod - 2)
    #     return pow2, logpow2, invpow2
    #
    #
    # def NTT(p, r, pow2, logpow2, invpow2, forward):
    #     j = pow2 // 2
    #     for i in range(1, pow2 - 1):
    #         if i >= j:
    #             p[i], p[j] = p[j], p[i]
    #         k = pow2 // 2
    #         while True:
    #             if k > j:
    #                 break
    #             j -= k
    #             k //= 2
    #         j += k
    #     l = 2
    #     m = len(r) * forward
    #     for _ in range(logpow2):
    #         for j in range(l // 2):
    #             for k in range(j, pow2, l):
    #                 a = p[k]
    #                 b = r[j * m // l] * p[k + l // 2]
    #                 b %= mod
    #                 p[k] = (a + b) % mod
    #                 p[k + l // 2] = (a - b) % mod
    #         l *= 2
    #     if forward == -1:
    #         for i in range(pow2):
    #             p[i] *= invpow2
    #             p[i] %= mod
    #
    #
    # def get_answer(c, v, n, s, k):
    #     pow2, logpow2, invpow2 = get_pow2(2 * k)
    #     prim_root = power(3, (mod - 1) // pow2)
    #     r = [1] * pow2
    #     for i in range(1, pow2):
    #         r[i] = r[i - 1] * prim_root
    #         r[i] %= mod
    #     factorial = [1] * pow2
    #     factorial_inv = [1] * pow2
    #     for i in range(1, pow2):
    #         factorial[i] = factorial[i - 1] * i
    #         factorial[i] %= mod
    #         factorial_inv[i] = power(factorial[i], mod - 2)
    #     vpowers = [[1] * pow2 for _ in range(n + 1)]
    #     for i in range(1, n + 1):
    #         for l in range(1, pow2):
    #             vpowers[i][l] = vpowers[i][l - 1] * v[i]
    #             vpowers[i][l] %= mod
    #     dp = [[0] * (k + 1) for _ in range(s + 1)]
    #     dp[0][0] = 1
    #     p1 = [0] * pow2
    #     p2 = [0] * pow2
    #     for i in range(1, n + 1):
    #         for j in range(s, c[i] - 1, -1):
    #             for l in range(pow2):
    #                 if l <= k:
    #                     p1[l] = dp[j - c[i]][l] * factorial_inv[l] % mod
    #                     p2[l] = factorial_inv[l] * vpowers[i][l] % mod
    #                 else:
    #                     p1[l] = 0
    #                     p2[l] = 0
    #             NTT(p1, r, pow2, logpow2, invpow2, 1)
    #             NTT(p2, r, pow2, logpow2, invpow2, 1)
    #             for l in range(pow2):
    #                 p1[l] *= p2[l]
    #                 p1[l] %= mod
    #             NTT(p1, r, pow2, logpow2, invpow2, -1)
    #             for l in range(k + 1):
    #                 dp[j][l] += p1[l] * factorial[l]
    #                 dp[j][l] %= mod
    #     answer = 0
    #     for j in range(s + 1):
    #         answer += dp[j][k]
    #         answer %= mod
    #     print(answer)
    #
    #
    # def solve():
    #     n, s, k = list(map(int, stdin.readline().strip().split()))
    #     c = [0] * (n + 1)
    #     v = [0] * (n + 1)
    #     for i in range(n):
    #         c[i + 1], v[i + 1] = list(map(int, stdin.readline().strip().split()))
    #     get_answer(c, v, n, s, k)
    #     stdin.close()
    #
    #
    # solve()



# from typing import List
# from collections import defaultdict
#
# class Solution:
#     def findJudge(self, N: int, trust: List[List[int]]) -> int:
#         """
#         like the find the celebrity
#         """
#         ingress = defaultdict(set)
#         egress =defaultdict(set)
#         for p, q in trust:
#             egress[p].add(q)
#             ingress[q].add(p)
#         for i in range(1, N+1):
#             if len(egress[i]) == 0 and len(ingress[i]) == N - 1:
#                 return i
#         return -1
# 2 , [1,2]

#
# # Function to rightRotate array
# def RightRotate(a, n, k):
#     # Input: Array[] = {1, 3, 5, 7, 9}, K = 2.
#     # If rotation is greater
#     # than size of array
#     k = k % n;
#
#     for i in range(0, n):
#
#         if (i < k):
#
#             # Printing rightmost
#             # kth elements
#             print(a[n + i - k], end=" ");
#
#         else:
#
#             # Prints array after
#             # 'k' elements
#             print(a[i - k], end=" ");
#
#     print("\n");
#
#
# # Driver code
# Array = [1, 2, 3, 4, 5];
# N = len(Array);
# K = 2;
#
# RightRotate(Array, N, K);

if __name__ == '__main__':

    stock = [225, 224, 407, 221, 259, 403]
