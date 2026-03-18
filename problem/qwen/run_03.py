# Qwen Attempt 3: Priority Queue Greedy Deletion (Fails)
import sys
import heapq

def solve():
    input = sys.stdin.read
    data = input().split()
    if not data:
        return
    n = int(data[0])
    A = [int(x) for x in data[1:]]
    
    # Keep track of active neighbors via linked list
    left = [i - 1 for i in range(n)]
    right = [i + 1 for i in range(n)]
    active = [True] * n
    
    # Priority queue to delete the largest valid elements first to minimize sum
    # Max heap (storing negative values)
    pq = []
    for i in range(1, n - 1):
        if A[i-1] % 2 != A[i+1] % 2:
            heapq.heappush(pq, (-A[i], i))
            
    while pq:
        val, idx = heapq.heappop(pq)
        val = -val
        if not active[idx]: continue
        
        l = left[idx]
        r = right[idx]
        
        # Verify conditions are still met
        if l >= 0 and r < n and A[l] % 2 != A[r] % 2:
            # Delete idx
            active[idx] = False
            
            # Update linked list
            if l >= 0: right[l] = r
            if r < n: left[r] = l
            
            # Re-evaluate l and r for possible deletions
            if l > 0 and l < n - 1:
                ll = left[l]
                rr = right[l]
                if ll >= 0 and rr < n and A[ll] % 2 != A[rr] % 2:
                    heapq.heappush(pq, (-A[l], l))
                    
            if r > 0 and r < n - 1:
                ll = left[r]
                rr = right[r]
                if ll >= 0 and rr < n and A[ll] % 2 != A[rr] % 2:
                    heapq.heappush(pq, (-A[r], r))
                    
    ans = [A[i] for i in range(n) if active[i]]
    print(f"{len(ans)} {sum(ans)}")

if __name__ == '__main__':
    solve()
