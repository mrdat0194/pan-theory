import cmath

def next_power_of_2(n: int) -> int:
    """
    Finds the next power of 2 greater than or equal to n.
    This is required because the classic Cooley-Tukey FFT algorithm 
    uses a divide-and-conquer approach that splits the array in half at each step.
    """
    power = 1
    while power < n:
        power *= 2
    return power

def fft(P: list[complex]) -> list[complex]:
    """
    Fast Fourier Transform (FFT) - Cooley-Tukey Algorithm.
    
    Mathematical Context:
    ---------------------
    A polynomial P(x) = a_0 + a_1*x + a_2*x^2 + ... + a_{n-1}*x^{n-1} is defined by its coefficients [a_0, a_1, ...].
    Evaluating this polynomial at n distinct points gives us a "point-value" representation.
    
    If we choose the points to be the n-th complex roots of unity (w_n^k = e^(2 * pi * i * k / n)),
    we can evaluate the polynomial in O(n log n) time instead of the naive O(n^2) time.
    
    The algorithm works by splitting the polynomial into Even and Odd degree terms:
    P(x) = P_even(x^2) + x * P_odd(x^2)
    
    Args:
        P: A list of complex numbers representing the coefficients of the polynomial.
           The length of P MUST be a power of 2. (Pad with zeros if necessary).
           
    Returns:
        A list of complex numbers representing the polynomial evaluated at the n-th roots of unity.
    """
    n = len(P)
    
    # Base case: A polynomial of degree 0 (a constant) evaluated at any point is just the constant itself.
    if n == 1:
        return P

    # 1. Divide: Separate the coefficients into even-indexed and odd-indexed parts.
    # P_even represents the polynomial a_0 + a_2*x + a_4*x^2 ...
    # P_odd represents the polynomial a_1 + a_3*x + a_5*x^2 ...
    P_even = P[0::2]
    P_odd = P[1::2]

    # 2. Conquer: Recursively evaluate the even and odd polynomials at the (n/2)-th roots of unity.
    y_even = fft(P_even)
    y_odd = fft(P_odd)

    # 3. Combine: Reconstruct the evaluations for the n-th roots of unity.
    y = [0] * n
    
    # The principal n-th root of unity: e^(2 * pi * i / n)
    omega_n = cmath.exp(2j * cmath.pi / n)
    
    # We start at omega_n^0 = 1
    omega = 1.0 + 0.0j

    # We only need to iterate up to n/2 because of the periodic properties of roots of unity:
    # omega_n^(k + n/2) = -omega_n^k
    for k in range(n // 2):
        # P(w^k) = P_even(w^(2k)) + w^k * P_odd(w^(2k))
        y[k] = y_even[k] + omega * y_odd[k]
        
        # P(w^(k + n/2)) = P_even(w^(2k)) - w^k * P_odd(w^(2k))
        y[k + n // 2] = y_even[k] - omega * y_odd[k]
        
        # Move to the next root of unity: w^(k+1)
        omega = omega * omega_n

    return y

def ifft(y: list[complex]) -> list[complex]:
    """
    Inverse Fast Fourier Transform (IFFT) / Polynomial Interpolation.
    
    Mathematical Context:
    ---------------------
    If we have the point-value representation of a polynomial (evaluated at the roots of unity),
    we want to recover the original coefficients [a_0, a_1, ...].
    
    This is mathematically equivalent to multiplying the values by the inverse of the Vandermonde matrix.
    Conveniently, the inverse of this matrix is just the original matrix but with the complex conjugate
    of the roots of unity, divided by n.
    
    Therefore, IFFT is just FFT evaluated with omega^(-1) instead of omega, scaled by 1/n.
    """
    n = len(y)
    
    # Base case
    if n == 1:
        return y

    y_even = y[0::2]
    y_odd = y[1::2]

    a_even = ifft(y_even)
    a_odd = ifft(y_odd)

    a = [0] * n
    
    # Key Difference: We use -2j to get the complex conjugate root of unity (omega^(-1))
    omega_n = cmath.exp(-2j * cmath.pi / n)
    omega = 1.0 + 0.0j

    for k in range(n // 2):
        a[k] = a_even[k] + omega * a_odd[k]
        a[k + n // 2] = a_even[k] - omega * a_odd[k]
        omega = omega * omega_n

    # The result must be divided by n at the very end of the recursion tree.
    # To avoid dividing by n at every recursive step, we only scale if this is the top-level call.
    # Wait, the easiest way to scale is actually to write a wrapper or scale at the end.
    # We will let a wrapper function handle the scaling to keep the recursion clean.
    return a

def interpolate_polynomial(y: list[complex]) -> list[complex]:
    """
    Wrapper for IFFT to scale the coefficients by 1/n.
    """
    n = len(y)
    unscaled_coeffs = ifft(y)
    return [c / n for c in unscaled_coeffs]

def multiply_polynomials(A: list[float], B: list[float]) -> list[float]:
    """
    Multiply two polynomials A(x) and B(x) in O(n log n) time using FFT.
    
    Steps:
    1. Pad A and B with zeros to length 2^k >= degree(A) + degree(B) + 1.
    2. Evaluate A and B at the roots of unity using FFT.
    3. Multiply the point values pairwise.
    4. Interpolate the result back to coefficients using IFFT.
    """
    # 1. Determine the required size for the resulting polynomial.
    # The degree of C(x) = A(x)B(x) is degree(A) + degree(B).
    # Thus, it has degree(A) + degree(B) + 1 coefficients.
    required_size = len(A) + len(B) - 1
    
    # Find the next power of 2 for FFT
    n = next_power_of_2(required_size)
    
    # Pad the coefficient lists with zeros up to size n
    A_padded = [complex(a, 0) for a in A] + [0j] * (n - len(A))
    B_padded = [complex(b, 0) for b in B] + [0j] * (n - len(B))
    
    # 2. Evaluate both polynomials at the n-th roots of unity (FFT)
    # This transforms them from Coefficient Representation -> Point-Value Representation
    A_values = fft(A_padded)
    B_values = fft(B_padded)
    
    # 3. Pairwise point multiplication
    # C(x) = A(x) * B(x). Therefore, for any root w, C(w) = A(w) * B(w)
    C_values = [A_values[i] * B_values[i] for i in range(n)]
    
    # 4. Interpolate back to coefficients (IFFT)
    # This transforms from Point-Value Representation -> Coefficient Representation
    C_coeffs_complex = interpolate_polynomial(C_values)
    
    # The resulting coefficients should be purely real. We take the real part and 
    # round off tiny floating point errors.
    C_coeffs = [round(c.real, 5) for c in C_coeffs_complex]
    
    # Trim trailing zeros that were added due to power-of-2 padding
    while len(C_coeffs) > 1 and C_coeffs[-1] == 0.0:
        C_coeffs.pop()
        
    return C_coeffs

def naive_multiply(A: list[float], B: list[float]) -> list[float]:
    """
    Standard O(n^2) polynomial multiplication for comparison.
    """
    m, n = len(A), len(B)
    result = [0.0] * (m + n - 1)
    for i in range(m):
        for j in range(n):
            result[i+j] += A[i] * B[j]
    return [round(x, 5) for x in result]

if __name__ == "__main__":
    print("="*60)
    print(" EDUCATIONAL FAST FOURIER TRANSFORM (FFT) DEMONSTRATION")
    print("="*60)
    
    # Let A(x) = 1 + 2x + 3x^2
    # Let B(x) = 4 + 5x + 6x^2
    A = [1, 2, 3]
    B = [4, 5, 6]
    
    print(f"\nPolynomial A(x): {A} (a_0 + a_1*x + ...)")
    print(f"Polynomial B(x): {B}")
    
    # 1. Naive O(n^2) Multiplication
    print("\n--- 1. Standard O(n^2) Convolution ---")
    C_naive = naive_multiply(A, B)
    print(f"C(x) = A(x) * B(x) = {C_naive}")
    
    # 2. O(n log n) FFT Multiplication
    print("\n--- 2. Fast O(n log n) FFT Convolution ---")
    C_fft = multiply_polynomials(A, B)
    print(f"C(x) = A(x) * B(x) = {C_fft}")
    
    print("\nVerification:")
    if C_naive == C_fft:
        print(" SUCCESS: FFT multiplication exactly matches naive multiplication!")
    else:
        print(" ERROR: Mismatch between FFT and naive methods.")
