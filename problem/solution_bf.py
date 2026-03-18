import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    if not data:
        return
    n = int(data[0])
    A = [int(x) for x in data[1:]]
    
    ans = set()
    def dfs(cur):
        ans.add(tuple(cur))
        for i in range(1, len(cur)-1):
            if cur[i-1] % 2 != cur[i+1] % 2:
                dfs(cur[:i] + cur[i+1:])
                
    dfs(A)
    min_l = min(len(x) for x in ans)
    min_s = min(sum(x) for x in ans if len(x) == min_l)
    
    print(f"{min_l} {min_s}")

if __name__ == '__main__':
    # Only suitable for N <= 15
    solve()
