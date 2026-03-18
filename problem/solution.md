# Solution Explanation

## The Invariant Property

The most crucial observation is that every valid deletion operation **never changes the total number of parity transitions** in the array. 

Let the elements be $A_1, A_2, \ldots, A_N$.
A parity transition occurs at index $i$ if $A_i \pmod 2 \neq A_{i+1} \pmod 2$. 

The deletion rule is: remove $A_i$ if $A_{i-1} \pmod 2 \neq A_{i+1} \pmod 2$.
If we delete $A_i$, the subarray $A_{i-1}, A_i, A_{i+1}$ (which has exactly 1 parity transition regardless of the parity of $A_i$) collapses into $A_{i-1}, A_{i+1}$. Because they have different parities, this new adjacent pair $A_{i-1}, A_{i+1}$ has exactly 1 parity transition.
Therefore, the number of parity transitions $T$ is invariant for the entire sequence of operations.

Since the number of transitions never changes, an array with $T$ parity transitions must have a minimum length of exactly $T+1$ (assuming $T > 0$). If $T = 0$, then no two adjacent elements have different parities, meaning no element can ever be removed, and the final length is $N$.

## Computing the Minimal Sum

Since the minimum length is exactly $T+1$ and the final array has $T$ parity transitions, the final array's elements must form a strictly alternating sequence of parities.
We can partition the initial array into contiguous blocks of identical parities. There will be exactly $T+1$ such blocks.

Let the blocks be $B_1, B_2, \ldots, B_{T+1}$.
To form the minimal length alternating array, we must pick exactly one element from each of these $T+1$ blocks.

By the rules of the problem:
1. We cannot remove the very first element ($A_1$) or the very last element ($A_N$). Thus, the chosen element from $B_1$ must be $A_1$, and the chosen element from $B_{T+1}$ must be $A_N$.
2. For all intermediate blocks $B_i$ ($1 < i < T+1$), we can independently choose to leave ANY single element while deleting the rest.

To minimize the sum of the remaining array, we should greedily pick the minimum element from each intermediate block $B_i$.

## Final Algorithm

1. Scan the array and detect blocks of identical parity. Let's denote them as $B_1, B_2, \dots B_K$.
2. If $K=1$, print $N$ and the sum of all elements in the array.
3. If $K > 1$, print the minimum length $K$.
4. The minimum possible sum will be exactly:
$A_1 + \left( \sum_{i=2}^{K-1} \min(B_i) \right) + A_N$

## Complexity
- **Time Complexity**: $O(N)$ since finding parity blocks and their minimums requires a single pass.
- **Space Complexity**: $O(1)$ auxiliary or $O(N)$ if storing blocks strictly. Perfect for limits.
