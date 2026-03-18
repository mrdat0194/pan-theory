# Qwen Attempt 2: Interval DP Approach (Fails due to TLE / MLE)
import sys

def solve():
    # DP solution
    input = sys.stdin.read
    data = input().split()
    if not data:
        return
    n = int(data[0])
    A = [int(x) for x in data[1:]]
    
    # DP[i][j] stores the minimum length and minimum sum to reduce A[i...j]
    # This leads to O(N^3) Time and O(N^2) Space scaling which gets TLE/MLE for N=10^5
    
    if n > 1000:
        # Fallback to greedy if N is too large (which leads to WA)
        ans = [A[0], A[-1]]
        print(f"2 {sum(ans)}")
        return
        
    dp_len = [[0] * n for _ in range(n)]
    dp_sum = [[0] * n for _ in range(n)]
    
    for i in range(n):
        dp_len[i][i] = 1
        dp_sum[i][i] = A[i]
        
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            # Base logic...
            dp_len[i][j] = length 
            dp_sum[i][j] = sum(A[i:j+1])
            # Fails to correctly model the parity dependencies of inner collapses
            
    print(f"{dp_len[0][n-1]} {dp_sum[0][n-1]}")

if __name__ == '__main__':
    solve()
