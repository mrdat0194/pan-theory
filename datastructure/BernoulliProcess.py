# pip install stochastic
from stochastic.processes.discrete import BernoulliProcess

bp = BernoulliProcess(p=0.5)

count = 0
n = 9000

for i in range(n):
    s = bp.sample(10)
    if sum(s) == 6 or sum(s) == 7 or sum(s) == 8:
        count += 1

print(count / n)

import operator as op
from functools import reduce


def ncr(n, r):
    r = min(r, n - r)
    if r == 0: return 1
    numer = reduce(op.mul, range(n, n - r, -1))
    denom = reduce(op.mul, range(1, r + 1))
    return numer // denom


# n tries, sth happens k times with a chance of p
def bernoulli_trial(n, k, p):
    return ncr(n, k) * (p ** k) * ((1 - p) ** (n - k))


# n tries, sth happens k or less than k times with a chance of p
def bernoulli_process(n, k, p):
    b = 0
    for i in range(0, k + 1):
        b = b + ncr(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return b


# n tries, sth happens between k1 and k2 (inclusive) times with a chance of p
def bernoulli_process_btw(n, k1, k2, p):
    b = 0
    for i in range(k1, k2 + 1):
        b = b + ncr(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return b


# returns the inverse of x
def inverse(x):
    if x == 0: return False
    return x ** (-1)


z = bernoulli_process_btw(10, 6, 8, 0.5)
print(z)