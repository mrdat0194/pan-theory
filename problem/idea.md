# Problem Idea Development

## The Initial Concept
The goal was to create a Div1 A / Div2 C problem that forces an LLM to fail completely on standard algorithmic tests (e.g., trying DP, greedy stacks, two-pointers, or interval DP) while being solvable in $O(N)$ with a clever mathematical observation.

I started by brainstorming constraints on subsegment deletions. Classic problems usually ask "can you reduce the string to empty?" and typically involve parity or bracket-matching variations that are well known.

To make it unique and search-proof, I arrived at a deletion rule that is completely dependent on local dynamic states, specifically the parity of the surviving neighbors:
`Remove element at index i if A[i-1] % 2 != A[i+1] % 2`

## The Invariant Observation
A naive approach would be to simulate removals or write an $O(N^3)$ interval dynamic programming solution to find state reductions. However, analyzing the properties of the operation reveals a powerful invariant: **The number of parity transitions in the array (where $A[i] \bmod 2 \neq A[i+1] \bmod 2$) never changes!**

Let's test this locally. If a sequence is $A, X, B$ where $A$ and $B$ have different parities (meaning the transition condition is met to delete $X$), the number of transitions in substring $A, X, B$ is EXACTLY 1, regardless of $X$'s parity. When $X$ is deleted, we are left with $A, B$, which ALSO has EXACTLY 1 transition because $A$ and $B$ have different parities. Thus, a local operation preserves the global number of parity transitions perfectly.

## Refining the Task
If the task just asked for minimum length, it would be a Div2 A level observation: length is simply $T+1$ (where $T=$ number of parity transitions) or $N$ if $T=0$.
To make it harder and prevent LLMs from guessing the simplest output, I added a secondary optimization: **"Minimize the length, AND amongst optimal lengths, minimize the sum of the remaining array elements!"**

This forces the contestant (or LLM) to deduce the explicit form of the remaining elements. Since the sequence of parities cannot change and length is exactly $T+1$, the array can be partitioned into $T+1$ blocks of identical parities. The final array must be constructed by picking exactly one element from each block.
Since the boundary elements $A_1$ and $A_N$ cannot be removed, the problem fully decouples: 
- Pick $A_1$ from block $1$ 
- Pick $\min(B_i)$ from each intermediate block $i$
- Pick $A_N$ from block $T+1$

This makes the problem entirely about finding invariants, completely derailing traditional problem-solving patterns algorithms. LLMs that try a Stack, greedy sweep, or interval DP generally fall into the sinkhole of false transitions.
