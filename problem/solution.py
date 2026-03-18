import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    if not data:
        return
    n = int(data[0])
    A = [int(x) for x in data[1:]]
    
    transitions = 0
    for i in range(1, n):
        if A[i] % 2 != A[i-1] % 2:
            transitions += 1
            
    if transitions == 0:
        print(f"{n} {sum(A)}")
        return
        
    blocks = []
    current_block = [A[0]]
    for i in range(1, n):
        if A[i] % 2 == A[i-1] % 2:
            current_block.append(A[i])
        else:
            blocks.append(current_block)
            current_block = [A[i]]
    blocks.append(current_block)
    
    min_sum = A[0] + A[-1]
    for i in range(1, len(blocks) - 1):
        min_sum += min(blocks[i])
        
    print(f"{transitions + 1} {min_sum}")

if __name__ == '__main__':
    solve()
