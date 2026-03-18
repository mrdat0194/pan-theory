# Qwen Attempt 1: Greedy Stack Approach (Fails)
import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    if not data:
        return
    n = int(data[0])
    A = [int(x) for x in data[1:]]
    
    # Greedy approach using a stack
    stack = []
    for num in A:
        stack.append(num)
        # Try to reduce if possible
        while len(stack) >= 3:
            if stack[-3] % 2 != stack[-1] % 2:
                # Remove the middle element
                removed = stack.pop(-2)
            else:
                break
                
    print(f"{len(stack)} {sum(stack)}")

if __name__ == '__main__':
    solve()
