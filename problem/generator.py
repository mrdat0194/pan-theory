import os
import random

def solve_optimal(A):
    n = len(A)
    transitions = 0
    for i in range(1, n):
        if A[i] % 2 != A[i-1] % 2:
            transitions += 1
            
    if transitions == 0:
        return f"{n} {sum(A)}\n"
        
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
        
    return f"{transitions + 1} {min_sum}\n"

def generate_test_cases():
    os.makedirs('test_cases', exist_ok=True)
    
    # Test 1: Small normal case
    t1 = [2, 4, 3, 5, 7, 2, 8, 1]
    _write_test(1, t1)
    
    # Test 2: No transitions array
    t2 = [1, 3, 5, 7]
    _write_test(2, t2)
    
    # Test 3: Large alternating array
    t3 = [random.randint(1, 1000) * 2 + (i % 2) for i in range(100)]
    _write_test(3, t3)
    
    # Test 4: Maximum size edge case (many large blocks)
    A = []
    parity = 0
    for i in range(1000):
        length = random.randint(1, 10)
        for _ in range(length):
            A.append(random.randint(1, 1000) * 2 + parity)
        parity ^= 1
    _write_test(4, A)
    
    # Test 5: Only two blocks
    A = []
    for _ in range(500): A.append(random.randint(1, 100) * 2)
    for _ in range(500): A.append(random.randint(1, 100) * 2 + 1)
    _write_test(5, A)

def _write_test(idx, A):
    with open(f'test_cases/{idx}.in', 'w') as f:
        f.write(f"{len(A)}\n")
        f.write(" ".join(map(str, A)) + "\n")
        
    with open(f'test_cases/{idx}.out', 'w') as f:
        f.write(solve_optimal(A))

if __name__ == '__main__':
    generate_test_cases()
