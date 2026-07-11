# Linear Algebra Course Curriculum Guide

Welcome to the Linear Algebra Course! This guide provides a logical, step-by-step curriculum to help you navigate the Jupyter notebooks and master the concepts, from basic vector arithmetic to advanced decompositions like SVD and Principal Component Analysis (PCA).

---

## Companion Materials & Author Resources

This curriculum is designed to align with Dr. Mike X Cohen's textbook **"Linear Algebra: Theory, Intuition, Code"** and his online resources.

- **Official Website:** [Sincxpress Books](https://sincxpress.com/books.html) — Official portal to purchase the high-quality PDF or paperback edition of the textbook directly from the author.
- **Code Repositories:**
  - [mikexcohen/LinAlgCourse](https://github.com/mikexcohen/LinAlgCourse) — The course repository matching the organized folders here.
  - [mikexcohen/LinAlgBook](https://github.com/mikexcohen/LinAlgBook) — The textbook's companion code repository.
- **Video Courses:** Visit [sincxpress.com](https://sincxpress.com) for video crash-courses and his full Udemy Linear Algebra masterclass.

> [!NOTE]
> Dr. Mike X Cohen's books are copyrighted. To support high-quality STEM education and ensure you have the correct, complete content, please purchase the official PDF directly from [Sincxpress Books](https://sincxpress.com/books.html).

---

## Recommended Learning Path

### Phase 1: Vector Foundations
*Corresponding Textbook Chapters: Chapter 2 (Vectors), Chapter 3 (Vector Multiplication)*
Start here to build the geometric and algebraic intuition that underlies all of linear algebra.
1. **[linalg_vectors](file:///C:/Users/mrdat/PycharmProjects/pan-theory/datastructure/Lesson/Linalg/linalg_vectors)**
   - Start with basic vector operations: addition, subtraction, scalar multiplication.
   - Master the **Dot Product** (algebraic and geometric formulations).
   - Learn about unit vectors, span, linear independence, and the cross product.

### Phase 2: Matrices and Multiplications
*Corresponding Textbook Chapters: Chapter 5 (Matrices), Chapter 6 (Matrix Multiplication)*
Transition from vectors to 2D arrays of numbers (matrices) and how they interact.
2. **[linalg_matrixmults](file:///C:/Users/mrdat/PycharmProjects/pan-theory/datastructure/Lesson/Linalg/linalg_matrixmults)**
   - Learn standard matrix multiplication, order of operations, and transposition.
   - Study properties of symmetric matrices.
   - Understand vector and matrix norms (Frobenius norm).

### Phase 3: Systems of Equations, Spaces, and Rank
*Corresponding Textbook Chapters: Chapter 4 (Vector Spaces), Chapter 7 (Rank), Chapter 8 (Matrix Spaces), Chapter 10 (Systems of Equations)*
Learn how matrices represent linear transformations and systems of equations.
3. **[linalg_systems](file:///C:/Users/mrdat/PycharmProjects/pan-theory/datastructure/Lesson/Linalg/linalg_systems)**
   - Solve systems of linear equations using Row Reduction (RREF).
4. **[linalg_rank](file:///C:/Users/mrdat/PycharmProjects/pan-theory/datastructure/Lesson/Linalg/linalg_rank)**
   - Understand the concept of matrix rank (dimension of column/row space).
   - Learn about rank-shifting (regularization).
5. **[linalg_spaces](file:///C:/Users/mrdat/PycharmProjects/pan-theory/datastructure/Lesson/Linalg/linalg_spaces)**
   - Explore vector spaces, subspaces, null space, and column space.

### Phase 4: Determinants, Inverses, and Projection
*Corresponding Textbook Chapters: Chapter 11 (Determinants), Chapter 12 (Matrix Inverse), Chapter 13 (Projections), Chapter 14 (Least-Squares)*
Learn how to check if a matrix is invertible, how to invert it, and how to project vectors.
6. **[linalg_matrixdet](file:///C:/Users/mrdat/PycharmProjects/pan-theory/datastructure/Lesson/Linalg/linalg_matrixdet)**
   - Learn properties of determinants and how they scale space.
7. **[linalg_inverse](file:///C:/Users/mrdat/PycharmProjects/pan-theory/datastructure/Lesson/Linalg/linalg_inverse)**
   - Calculate matrix inverses, left/right inverses for non-square matrices, and the Moore-Penrose Pseudoinverse.
8. **[linalg_projorth](file:///C:/Users/mrdat/PycharmProjects/pan-theory/datastructure/Lesson/Linalg/linalg_projorth)**
   - Project vectors in $\mathbb{R}^2$ and $\mathbb{R}^N$.
   - Understand Orthogonality, Gram-Schmidt orthogonalization, and the QR Decomposition.
9. **[linalg_leastsquares](file:///C:/Users/mrdat/PycharmProjects/pan-theory/datastructure/Lesson/Linalg/linalg_leastsquares)**
   - Use projections to solve overdetermined systems (ordinary least squares regression).

### Phase 5: Spectral Theory and Dimensionality Reduction
*Corresponding Textbook Chapters: Chapter 15 (Eigendecomposition), Chapter 16 (Singular Value Decomposition (SVD)), Chapter 17 (Quadratic Form), Chapter 18 (Covariance Matrices), Chapter 19 (Principal Components Analysis (PCA))*
The most advanced and widely applied concepts in data science and machine learning.
10. **[linalg_eig](file:///C:/Users/mrdat/PycharmProjects/pan-theory/datastructure/Lesson/Linalg/linalg_eig)**
    - Find eigenvalues and eigenvectors.
    - Diagonalize matrices and study eigenvalues of symmetric matrices.
11. **[linalg_quadformdefinite](file:///C:/Users/mrdat/PycharmProjects/pan-theory/datastructure/Lesson/Linalg/linalg_quadformdefinite)**
    - Learn quadratic forms, definiteness (positive/negative definite), and their geometry.
12. **[linalg_svd](file:///C:/Users/mrdat/PycharmProjects/pan-theory/datastructure/Lesson/Linalg/linalg_svd)**
    - Learn Singular Value Decomposition (SVD), the pinnacle decomposition of linear algebra.
13. **[linalg_spectralsvd](file:///C:/Users/mrdat/PycharmProjects/pan-theory/datastructure/Lesson/Linalg/linalg_spectralsvd)**
    - Apply SVD to low-rank approximations and Principal Component Analysis (PCA) for dimensionality reduction.
