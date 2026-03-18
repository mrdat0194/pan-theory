# Alternating Parity Deletions

## Problem Statement

You are given an array $A$ of $N$ positive integers.

You are allowed to perform the following operation any number of times (possibly zero):
*   Select an index $i$ ($2 \le i \le |A| - 1$) such that $A_{i-1} \pmod 2 \neq A_{i+1} \pmod 2$.
*   Remove $A_i$ from the array. The remaining parts of the array are concatenated without changing their relative order.

Note that $A_1$ (the first element) and $A_{|A|}$ (the last element) can never be removed since they lack an adjacent left or right element, respectively.

Your goal is to apply operations to minimize the length of the remaining array. Once you find the minimum possible length of the resulting array, among all possible sequences of operations that achieve this minimum length, you must find the one that also minimizes the sum of the elements in the remaining array.

## Input

The first line of the input contains a single integer $N$ ($1 \le N \le 10^5$) — the number of elements in the initial array $A$.
The second line contains $N$ space-separated integers $A_1, A_2, \dots, A_N$ ($1 \le A_i \le 10^9$) — the elements of the array.

## Output

Print two space-separated integers: 
1. The minimum possible length of the final array.
2. The minimum possible sum of its elements among all final arrays of that minimum length.

## Examples

**Input 1:**
```
8
2 4 3 5 7 2 8 1
```

**Output 1:**
```
4 8
```

**Input 2:**
```
4
1 3 5 7
```

**Output 2:**
```
4 16
```

## Note
In the first example, the initial parities are `0 0 1 1 1 0 0 1`.
One optimal sequence of deletions is to remove $A_2$ (4), $A_5$ (7), and $A_4$ (5). The remaining elements are $2, 3, 2, 1$. The length is $4$ and the sum is $2+3+2+1=8$.
In the second example, no element satisfies the deletion condition because all elements are odd, meaning any element's neighbors are odd, thus not having different parities. We can delete nothing.
