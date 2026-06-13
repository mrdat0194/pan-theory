#!/bin/python3
# https://www.hackerrank.com/contests/w22/challenges/number-of-sequences/

'''
So for those who are confused, here's an explanation of the problem that makes sense for me (and hopefully will help you too):

Disclaimer: I have not come up with a complete solution yet, this is just intended to help clarify the problem statement for anyone who is confused.

Here's an example sequence:

k = 1 2 3 4 5 6 ak= -1 -1 -1 -1 -1 -1

So to satisfy the first condition,

a1 = 0; a2 = 0,1; a3 = 0,1,2; a4 = 0, ...etc.

The second condition is trickier... It says "ak = am % k for k,m such that k divides m". It may help to understand it better as k is a divisor of m or that m is a multiple of k. In other words, the value of a2 will affect which values are possible for a4,a6,a8 and so on because 4,6,8 are multiples of 2.

Here's the logic behind satisfying the second condition:

a1: must be 0; a2: can be 0 or 1, so it can be 2 total things a3: is not a multiple of 2 (1 has no effect) so there are 3 possible values of 0,1,2 a4: IS A MULTIPLE OF 2! So this means that the value of a2 = a4 % 2 because k = 2 divides m = 4. Moreover, 0 <= a4 <= 3. So, thinking backwards, if a2 = 0 then a4%2 = 0. This is only true when a4 = 0,2. if a2 = 1 then a4%2 = 1. This is true when a4 = 1,3. So if you were to write out all possible combinations of a2 and a4 it would look like this: (0,0),(0,2),(1,1),(1,3) for a total of 4 combinations, doubling the number of combinations possible with just a2. In other words, regardless of the value of a2 you will always have 2 different options for 4. a5: is prime, so it is not a multiple of a1-a4. Therefore only the first constraint applies, 0<=a5<=5-1 = 4 a6: is divisible by 2 and 3. so apply the same principle as with a4 only with 2 variables instead of 1. There should be only 6 possible combinations of 2,3, and 6 that work. There are 2 options for a2 and 3 options for a3, so there is no net contribution from a6. In other words, it is fixed based on what values are assigned to a2 and a3.

So in total you have: a1: 1 option; a2: 2 options; a3: 3 options; a4: 2 options; (2 for each of the options in a2) a5: 5 options; a6: 1 option; (each value of a6 predetermined by a2,a3)

1*2*3*2*5*1 = 60 possible 'nice' sequences
'''

from math import factorial


def prime_factors(n):
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
    """ Returns  a list of primes < n """
    sieve = [True] * (n // 2)
    for i in range(3, int(n ** 0.5) + 1, 2):
        if sieve[i // 2]:
            sieve[i * i // 2::i] = [False] * ((n - i * i - 1) // (2 * i) + 1)
    return [2] + [2 * i + 1 for i in range(1, n // 2) if sieve[i]]


def number_nice(A):
    if A[0] != 0 and A[0] != -1:
        return 0
    # elif len(A)==2
    else:
        nice_seqs = 1
        N = len(A)
        # a[1] = 0 always
        # a[2] = 0 or 1
        for prime in primes(N + 1):
            # print(prime)
            positions = [i for i in range(N) if
                         (i + 1) % prime == 0]  # all positions at multiples of prime in A, 0-indexed
            test = [A[i] for i in positions]
            # print(test)
            fixed_mods = [x % prime for x in test if x != -1]

            if len(set(fixed_mods)) > 1:
                return 0  # given sequence contains contradiction, cannot be fixed

            elif -1 not in test:
                continue  # if nothing changeable and every element in sequence mod prime is equal, move to next prime

            elif -1 in test and len(set(test)) == 1:  # all -1's
                nice_seqs *= prime * factorial((positions[-1] + 1) / prime)

            else:  # mixture of -1's and elements with equal mods wrt prime
                the_remainder = fixed_mods[0]
                minus_ones = [i + 1 for i in positions if A[i] == -1]  # 1-indexed positions of -1's in test
                for position in minus_ones:
                    nice_seqs *= position / prime
    return nice_seqs


n = int(input().strip())
arr = [int(x) for x in input().strip().split(' ')]

print(number_nice(arr) % (7 + 10 ** 9))