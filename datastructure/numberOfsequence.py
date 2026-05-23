#!/bin/python3
# https://www.hackerrank.com/contests/w22/challenges/number-of-sequences/

'''
=============================================================================
PROBLEM: Number of Sequences  (HackerRank Week of Code 22)
https://www.hackerrank.com/contests/w22/challenges/number-of-sequences/
=============================================================================

PROBLEM STATEMENT (plain English)
-----------------------------------
You are given a sequence A of length N where each element A[k] is either a
known integer (0-indexed internally, but 1-indexed in the problem) or -1
(meaning "unknown / free to choose").

A sequence is called "nice" if ALL of these are true for every position k:

  Condition 1 (range):   0 <= a[k] <= k-1
                         i.e. a[k] can be at most k-1  (0-based: at most its index)

  Condition 2 (divisor): For every pair (k, m) where k divides m (k | m),
                         a[k] = a[m] % k
                         i.e. the value at position k CONSTRAINS every multiple
                         of k: the multiple's value, mod k, must equal a[k].

Count how many distinct "nice" sequences are consistent with the given A
(where -1 positions are free). Answer modulo 10^9 + 7.

=============================================================================
WORKED EXAMPLE  (N=6, all -1: A = [-1, -1, -1, -1, -1, -1])
=============================================================================

Step 1 — How many free choices does each position contribute?
-------------------------------------------------------------
Without Condition 2, position k has k choices: {0, 1, ..., k-1}.
But Condition 2 ties positions together via divisibility.

The KEY INSIGHT is:
  The value at prime power position p^e INDEPENDENTLY determines the "residue
  class" for ALL multiples of p^e.
  If a[p^e] is free (-1), you get exactly p new free choices — NOT p^e choices —
  because the value you pick for a[p^e] propagates as a constraint to all
  multiples of p^e (they must agree mod p^e).

Step 2 — Walk through positions 1..6:
--------------------------------------
  a[1] = 0  always (only value in {0,..,0}). Contributes factor of 1.

  a[2]: free in {0,1}. Choose it freely → 2 choices.
        This fixes: a[4] % 2 = a[2], a[6] % 2 = a[2].

  a[3]: free in {0,1,2}. Not a multiple of 2. → 3 choices.
        This fixes: a[6] % 3 = a[3].

  a[4]: p=2, e=2 (p^e = 4). If a[4] is free, how many extra choices?
        a[4] ∈ {0,1,2,3} BUT a[4] % 2 is already fixed by a[2].
        So only 4/2 = 2 residues in {0,1,2,3} satisfy each value of a[2]:
          if a[2]=0: a[4] ∈ {0,2}  (both give a[4]%2=0)
          if a[2]=1: a[4] ∈ {1,3}  (both give a[4]%2=1)
        → Factor of 2 (= p = 2) from a[4].

  a[5]: prime p=5. No divisors among {1,2,3,4} divide 5 (5 is prime).
        Only constraint is range: a[5] ∈ {0,1,2,3,4} → 5 free choices.
        → Factor of 5.

  a[6]: divisible by both 2 and 3. So a[6] % 2 = a[2] AND a[6] % 3 = a[3].
        By CRT, for any fixed (a[2], a[3]) there is EXACTLY ONE value in
        {0,..,5} that satisfies both constraints simultaneously.
        → Factor of 1 (a[6] is fully determined, no extra freedom).

Step 3 — Multiply the independent factors:
------------------------------------------
  a[1] × a[2] × a[3] × a[4] × a[5] × a[6]
  =  1  ×  2  ×  3  ×  2  ×  5  ×  1
  = 60 nice sequences  ✓

=============================================================================
ALGORITHM  (how the code works)
=============================================================================

For each prime p and each exponent e such that p^e <= N:
  - Look at all multiples of p^e in A (positions p^e, 2*p^e, 3*p^e, ...).
  - If any of those positions has a known value, compute val % p^e.
    All known multiples must agree on this residue; if they don't → return 0.
  - If NO multiple has a known value (all free), multiply the answer by p,
    because choosing a[p^e] freely gives exactly p new degrees of freedom.

Why multiply by p and not p^e?
  Each new prime-power level p^e adds exactly p choices on top of what p^(e-1)
  already constrains. The prime-power factorisation of each position's range
  [0, k-1] decomposes into independent contributions from each (p, e) pair.

Time complexity: O(N log N)  — same structure as the Sieve of Eratosthenes.
=============================================================================
'''

# from math import factorial


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
    N = len(A)
    MOD = 10**9 + 7
    
    if A[0] != 0 and A[0] != -1:
        return 0
    
    nice_seqs = 1
    for prime in primes(N + 1):
        e = 1
        while prime**e <= N:
            q = prime**e
            fixed_val = -1
            # Check all multiples of this prime power q = p^e
            for k in range(q, N + 1, q):
                if A[k-1] != -1:
                    val = A[k-1] % q
                    if fixed_val != -1 and fixed_val != val:
                        return 0  # Contradiction
                    fixed_val = val
            
            if fixed_val == -1:
                nice_seqs = (nice_seqs * prime) % MOD
            e += 1
            
    return nice_seqs


if __name__ == '__main__':
    n = int(input().strip())
    arr = [int(x) for x in input().strip().split(' ')]

    print(number_nice(arr))