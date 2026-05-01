# Linear Algebra for AI

Linear algebra is the mathematical backbone of AI. Neural networks, attention mechanisms, PCA, SVD, and optimization all reduce to matrix and vector operations.

## Table of Contents

1. [Vectors](#1-vectors)
   - 1.1 [Definition & Notation](#11-definition--notation)
   - 1.2 [Vector Operations](#12-vector-operations)
   - 1.3 [Dot Product](#13-dot-product)
   - 1.4 [Norms](#14-norms)
2. [Matrices](#2-matrices)
   - 2.1 [Definition & Notation](#21-definition--notation)
   - 2.2 [Matrix Operations](#22-matrix-operations)
   - 2.3 [Matrix Multiplication](#23-matrix-multiplication)
   - 2.4 [Special Matrices](#24-special-matrices)
   - 2.5 [Symmetric Matrix](#25-symmetric-matrix)
3. [Linear Transformations](#3-linear-transformations)
   - 3.1 [Geometric Intuition](#31-geometric-intuition)
   - 3.2 [Composition](#32-composition)
4. [Systems of Linear Equations](#4-systems-of-linear-equations)
   - 4.1 [Matrix Form](#41-matrix-form)
   - 4.2 [Invertibility](#42-invertibility)
5. [Determinant](#5-determinant)
   - 5.1 [Definition & Geometric Meaning](#51-definition--geometric-meaning)
   - 5.2 [Properties](#52-properties)
6. [Eigenvalues & Eigenvectors](#6-eigenvalues--eigenvectors)
   - 6.1 [Definition](#61-definition)
   - 6.2 [Why Eigenvectors Matter](#62-why-eigenvectors-matter)
   - 6.3 [Eigendecomposition](#63-eigendecomposition)
7. [Singular Value Decomposition (SVD)](#7-singular-value-decomposition-svd)
   - 7.1 [Definition](#71-definition)
   - 7.2 [Geometric Meaning](#72-geometric-meaning)
   - 7.3 [Low-Rank Approximation](#73-low-rank-approximation)
   - 7.4 [Why SVD Matters in AI](#74-why-svd-matters-in-ai)
8. [Vector Spaces & Subspaces](#8-vector-spaces--subspaces)
   - 8.1 [Linear Independence](#81-linear-independence)
   - 8.2 [Span, Basis & Dimension](#82-span-basis--dimension)
   - 8.3 [Rank & Null Space](#83-rank--null-space)
9. [Orthogonality](#9-orthogonality)
   - 9.1 [Orthogonal Vectors & Matrices](#91-orthogonal-vectors--matrices)
   - 9.2 [Projections](#92-projections)
10. [Positive Definite Matrices](#10-positive-definite-matrices)
11. [Gradients & Jacobians](#11-gradients--jacobians)
    - 11.1 [Gradient as a Vector](#111-gradient-as-a-vector)
    - 11.2 [Jacobian Matrix](#112-jacobian-matrix)
    - 11.3 [Hessian Matrix](#113-hessian-matrix)
12. [Matrix Derivatives](#12-matrix-derivatives)
    - 12.1 [Layout Convention](#121-layout-convention)
    - 12.2 [Scalar by Vector](#122-scalar-by-vector--partial-y--partial-mathbfx)
    - 12.3 [Scalar by Matrix](#123-scalar-by-matrix--partial-y--partial-w)
    - 12.4 [Vector by Vector — Jacobian](#124-vector-by-vector--jacobian-partial-mathbfy--partial-mathbfx)
    - 12.5 [Chain Rule in Matrix Form](#125-chain-rule-in-matrix-form)
    - 12.6 [Common Reference Table](#126-common-matrix-derivative-reference)
13. [AI Applications](#13-ai-applications)
    - 13.1 [Neural Network Forward Pass](#131-neural-network-forward-pass)
    - 13.2 [Attention Mechanism](#132-attention-mechanism)
    - 13.3 [PCA](#133-pca)

---

## 1. Vectors

### 1.1 Definition & Notation

A vector is an ordered list of numbers — a point or direction in $n$-dimensional space.

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n$$

| Symbol | Meaning |
|--------|---------|
| $\mathbf{x} \in \mathbb{R}^n$ | Column vector with $n$ real-valued entries |
| $\mathbf{x}^\top$ | Row vector — transpose of $\mathbf{x}$ |
| $x_i$ | The $i$-th element of $\mathbf{x}$ |

**In AI:** an input to a neural network is a vector — e.g., a 784-dim vector for a 28×28 image, or a 512-dim embedding for a word.

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0])   # shape (3,) — 1D vector
x = x.reshape(-1, 1)             # shape (3,1) — column vector
```

### 1.2 Vector Operations

**Addition and scalar multiplication:**

$$\mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \end{bmatrix}, \qquad c\mathbf{a} = \begin{bmatrix} ca_1 \\ ca_2 \end{bmatrix}$$

**Geometric meaning:** addition moves along one vector then another; scalar multiplication stretches or shrinks.

```python
a = np.array([1.0, 2.0])
b = np.array([3.0, 4.0])

a + b        # [4., 6.]
2 * a        # [2., 4.]
a - b        # [-2., -2.]
```

### 1.3 Dot Product

$$\mathbf{a} \cdot \mathbf{b} = \mathbf{a}^\top \mathbf{b} = \sum_{i=1}^n a_i b_i = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$$

| Term | Meaning |
|------|---------|
| $\sum a_i b_i$ | Element-wise multiply then sum |
| $\|\mathbf{a}\|\|\mathbf{b}\|$ | Product of the two lengths |
| $\cos\theta$ | Cosine of the angle between the vectors |
| Result is a **scalar** | Not a vector |

| Dot product value | Geometric meaning |
|-------------------|-------------------|
| $\mathbf{a} \cdot \mathbf{b} > 0$ | Vectors point in similar direction |
| $\mathbf{a} \cdot \mathbf{b} = 0$ | Vectors are perpendicular (orthogonal) |
| $\mathbf{a} \cdot \mathbf{b} < 0$ | Vectors point in opposite directions |

**In AI:** the dot product measures **similarity**. Attention scores, logits, and cosine similarity all use it.

```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

np.dot(a, b)      # 1×4 + 2×5 + 3×6 = 32.0
a @ b             # same — preferred in practice
```

### 1.4 Norms

A norm measures the **length (magnitude)** of a vector.

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2} \qquad (\text{L2 norm — Euclidean length})$$

$$\|\mathbf{x}\|_1 = \sum_{i=1}^n |x_i| \qquad (\text{L1 norm — Manhattan distance})$$

$$\|\mathbf{x}\|_\infty = \max_i |x_i| \qquad (\text{L}\infty\text{ norm — largest element})$$

| Norm | Used in AI for |
|------|----------------|
| L2 ($\|\cdot\|_2$) | Weight decay, gradient clipping, Euclidean distance |
| L1 ($\|\cdot\|_1$) | Sparse regularization (Lasso) |
| L$\infty$ | Adversarial robustness bounds |

```python
x = np.array([3.0, 4.0])

np.linalg.norm(x)      # L2: sqrt(9+16) = 5.0
np.linalg.norm(x, 1)   # L1: 3+4 = 7.0
np.linalg.norm(x, np.inf)  # Linf: max(3,4) = 4.0

# Normalize to unit vector
x_unit = x / np.linalg.norm(x)   # [0.6, 0.8]
```

---

## 2. Matrices

### 2.1 Definition & Notation

A matrix is a 2D array of numbers with $m$ rows and $n$ columns.

$$A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

| Symbol | Meaning |
|--------|---------|
| $A \in \mathbb{R}^{m \times n}$ | Matrix with $m$ rows and $n$ columns |
| $a_{ij}$ or $A_{ij}$ | Element at row $i$, column $j$ |
| $A^\top \in \mathbb{R}^{n \times m}$ | Transpose — rows become columns |

### 2.2 Matrix Operations

$$A + B: \quad (A+B)_{ij} = a_{ij} + b_{ij} \qquad \text{(same shape required)}$$

$$(A^\top)_{ij} = a_{ji} \qquad \text{(swap row and column indices)}$$

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

A + B          # [[6,8],[10,12]]
A.T            # [[1,3],[2,4]]   — transpose
2 * A          # [[2,4],[6,8]]   — scalar multiply
```

### 2.3 Matrix Multiplication

$$C = AB \in \mathbb{R}^{m \times p}, \quad A \in \mathbb{R}^{m \times n},\quad B \in \mathbb{R}^{n \times p}$$

$$c_{ij} = \sum_{k=1}^n a_{ik} b_{kj} = \text{(row } i \text{ of } A) \cdot \text{(column } j \text{ of } B)$$

**Shape rule:** inner dimensions must match — $(m \times \mathbf{n})(\mathbf{n} \times p) = (m \times p)$

```
A: (3 × 4)
B: (4 × 5)
C = AB: (3 × 5)   ← 4 matches 4 ✓
```

```python
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)

C = A @ B          # (3, 5)
C = np.matmul(A, B)  # same
```

**Matrix-vector multiply** is the core of a neural network layer:

$$\mathbf{y} = W\mathbf{x} + \mathbf{b}, \quad W \in \mathbb{R}^{m \times n},\ \mathbf{x} \in \mathbb{R}^n,\ \mathbf{y} \in \mathbb{R}^m$$

Each output $y_i$ is the dot product of row $i$ of $W$ with $\mathbf{x}$ — a weighted sum of all inputs.

### 2.4 Special Matrices

**Identity matrix $I$:** diagonal ones, zeros elsewhere. $AI = IA = A$.

$$I_3 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

**Diagonal matrix:** only diagonal entries are nonzero. Scales each dimension independently.

$$D = \begin{bmatrix} d_1 & 0 & 0 \\ 0 & d_2 & 0 \\ 0 & 0 & d_3 \end{bmatrix}, \qquad D\mathbf{x} = \begin{bmatrix} d_1 x_1 \\ d_2 x_2 \\ d_3 x_3 \end{bmatrix}$$

**Symmetric matrix:** $A^\top = A$, i.e., $a_{ij} = a_{ji}$. Covariance matrices are always symmetric.

**Orthogonal matrix:** $Q^\top Q = QQ^\top = I$, i.e., $Q^{-1} = Q^\top$. Preserves lengths and angles — pure rotation/reflection.

```python
np.eye(3)                          # 3×3 identity
np.diag([1.0, 2.0, 3.0])          # diagonal matrix
np.allclose(A, A.T)               # check symmetric
np.allclose(Q.T @ Q, np.eye(n))   # check orthogonal
```

### 2.5 Symmetric Matrix

A matrix is **symmetric** if it equals its own transpose:

$$A = A^\top \quad \Longleftrightarrow \quad a_{ij} = a_{ji} \text{ for all } i, j$$

**Example:**

$$A = \begin{bmatrix} 4 & 2 & 1 \\ 2 & 5 & 3 \\ 1 & 3 & 6 \end{bmatrix} \quad \leftarrow \text{symmetric: } a_{12}=a_{21}=2,\; a_{13}=a_{31}=1, \ldots$$

**Why symmetry matters — special properties:**

| Property | Meaning |
|----------|---------|
| All eigenvalues are **real** | No complex eigenvalues — safe to interpret |
| Eigenvectors are **orthogonal** | Different eigenvectors point in perpendicular directions |
| Always diagonalizable | $A = V\Lambda V^\top$ guaranteed |
| $\mathbf{x}^\top A \mathbf{y} = \mathbf{y}^\top A \mathbf{x}$ | Symmetric bilinear form |

**Where symmetric matrices appear in AI:**

| Matrix | Why symmetric |
|--------|--------------|
| Covariance matrix $\Sigma$ | $\text{Cov}(x_i, x_j) = \text{Cov}(x_j, x_i)$ by definition |
| Hessian $\nabla^2 f$ | Mixed partials are equal: $\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$ |
| Kernel matrix $K$ | $K_{ij} = k(x_i, x_j) = k(x_j, x_i)$ by symmetry of the kernel |
| Attention score matrix | $QK^\top$ is symmetric when $Q = K$ (self-attention) |
| Graph Laplacian $L$ | Encodes undirected graph structure |

```python
A = np.array([[4., 2., 1.],
              [2., 5., 3.],
              [1., 3., 6.]])

np.allclose(A, A.T)        # True — symmetric

# Symmetric matrix → use eigh (not eig): faster, guaranteed real, sorted
eigenvalues, eigenvectors = np.linalg.eigh(A)

# Verify orthogonality of eigenvectors
np.allclose(eigenvectors.T @ eigenvectors, np.eye(3))  # True
```

> **Key insight:** symmetric matrices are the "nicest" matrices in linear algebra — they have a complete set of real, orthogonal eigenvectors and are always diagonalizable.

---

## 3. Linear Transformations

### 3.1 Geometric Intuition

Every matrix $A \in \mathbb{R}^{m \times n}$ defines a linear transformation: it maps a vector in $\mathbb{R}^n$ to a vector in $\mathbb{R}^m$.

$$T(\mathbf{x}) = A\mathbf{x}$$

| Matrix type | Geometric effect |
|-------------|-----------------|
| Diagonal $D$ | Scales each axis independently |
| Orthogonal $Q$ | Rotates or reflects — preserves shape and size |
| Any $A$ | Combination of scaling, rotation, shearing |

**Example — 2D rotation by angle $\theta$:**

$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

```python
theta = np.pi / 4   # 45 degrees

R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

x = np.array([1.0, 0.0])   # unit vector pointing right
R @ x                       # [0.707, 0.707] — rotated 45°
```

### 3.2 Composition

Applying transformation $B$ then $A$ is equivalent to the single matrix $AB$:

$$A(B\mathbf{x}) = (AB)\mathbf{x}$$

**In AI:** a deep network with no activations is just one big matrix multiplication — that's why nonlinear activations (ReLU, Tanh) are essential. Without them, all layers collapse into a single linear transformation.

---

## 4. Systems of Linear Equations

### 4.1 Matrix Form

A system of $m$ equations with $n$ unknowns:

$$A\mathbf{x} = \mathbf{b}, \quad A \in \mathbb{R}^{m \times n},\ \mathbf{x} \in \mathbb{R}^n,\ \mathbf{b} \in \mathbb{R}^m$$

| Case | Meaning |
|------|---------|
| Unique solution | $A$ is invertible: $\mathbf{x} = A^{-1}\mathbf{b}$ |
| No solution | $\mathbf{b}$ outside the column space of $A$ |
| Infinite solutions | Underdetermined system — more unknowns than equations |

### 4.2 Invertibility

$A \in \mathbb{R}^{n \times n}$ is invertible if and only if:
- $\det(A) \neq 0$
- All eigenvalues are nonzero
- The columns are linearly independent (no column is a combination of the others)
- $\text{rank}(A) = n$

$$AA^{-1} = A^{-1}A = I$$

```python
A = np.array([[2.0, 1.0], [5.0, 3.0]])

np.linalg.det(A)      # 1.0   ← nonzero → invertible
A_inv = np.linalg.inv(A)

A @ A_inv             # ≈ identity
np.linalg.solve(A, b) # preferred over inv — more numerically stable
```

> **In practice:** never compute $A^{-1}$ explicitly. Use `np.linalg.solve(A, b)` — it is faster and numerically stable.

---

## 5. Determinant

### 5.1 Definition & Geometric Meaning

The **determinant** of a square matrix $A \in \mathbb{R}^{n \times n}$ is a single scalar that captures how the linear transformation $A$ scales volume.

$$\det(A) \in \mathbb{R}$$

**2×2 case:**

$$A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}, \qquad \det(A) = ad - bc$$

**Geometric meaning:** $|\det(A)|$ is the factor by which $A$ scales areas (2D) or volumes (3D).

```
Unit square:  area = 1
After A:      area = |det(A)|

det(A) = 2   →  area doubled
det(A) = 0.5 →  area halved
det(A) = 0   →  area = 0  (collapsed to a line or point)
det(A) < 0   →  orientation flipped (mirror image)
```

**Concrete 2D example:**

$$A = \begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix} \quad \Rightarrow \quad \det(A) = 3 \times 2 = 6$$

This matrix stretches the $x$-axis by 3 and the $y$-axis by 2, so a unit square becomes a $3 \times 2$ rectangle with area 6.

$$B = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix} \quad \Rightarrow \quad \det(B) = 1 \times 4 - 2 \times 2 = 0$$

Column 2 is exactly twice column 1 — the matrix collapses 2D space onto a line. No inverse exists.

### 5.2 Properties

| Property | Meaning |
|----------|---------|
| $\det(A) \neq 0$ | $A$ is invertible — no information is destroyed |
| $\det(A) = 0$ | $A$ is **singular** — collapses space, not invertible |
| $\det(I) = 1$ | Identity preserves volume |
| $\det(AB) = \det(A)\det(B)$ | Volume scaling composes multiplicatively |
| $\det(A^\top) = \det(A)$ | Transpose has same determinant |
| $\det(A^{-1}) = 1/\det(A)$ | Inverse undoes the scaling |
| $\det(cA) = c^n \det(A)$ | Scalar multiply scales det by $c^n$ |

**Connection to eigenvalues:**

$$\det(A) = \prod_{i=1}^n \lambda_i$$

The determinant is the **product of all eigenvalues**. If any eigenvalue is zero, the determinant is zero — meaning the matrix crushes at least one direction flat.

**Finding eigenvalues via determinant** — the characteristic equation:

$$\det(A - \lambda I) = 0$$

$$A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix} \quad \Rightarrow \quad \det\begin{bmatrix} 3-\lambda & 1 \\ 0 & 2-\lambda \end{bmatrix} = (3-\lambda)(2-\lambda) = 0 \quad \Rightarrow \quad \lambda = 3,\; 2$$

```python
A = np.array([[3.0, 1.0], [0.0, 2.0]])

np.linalg.det(A)          # 6.0 = 3 × 2

# det = product of eigenvalues
eigenvalues = np.linalg.eigvals(A)
np.prod(eigenvalues)      # 6.0 ✓

# Singular matrix
B = np.array([[1.0, 2.0], [2.0, 4.0]])
np.linalg.det(B)          # 0.0 — not invertible
```

> **One-line summary:** the determinant measures how much a matrix scales volume. Zero determinant = the matrix destroys information by collapsing space.

---

## 6. Eigenvalues & Eigenvectors

### 6.1 Definition

A nonzero vector $\mathbf{v}$ is an **eigenvector** of $A$ if multiplying by $A$ only **scales** it — the direction stays the same:

$$A\mathbf{v} = \lambda\mathbf{v}$$

| Term | Meaning |
|------|---------|
| $\mathbf{v}$ | **Eigenvector** — a special direction that $A$ does not rotate, only stretches/shrinks |
| $\lambda$ | **Eigenvalue** — the scalar factor by which $A$ scales along $\mathbf{v}$ |
| $\lambda > 1$ | Stretches in that direction |
| $0 < \lambda < 1$ | Shrinks in that direction |
| $\lambda < 0$ | Flips direction then scales |
| $\lambda = 0$ | Collapses that direction to zero (→ matrix is singular) |

**Geometric intuition:** most vectors get both rotated and scaled by $A$. Eigenvectors are the special ones that only get scaled — they are the "natural axes" along which $A$ acts.

**Concrete example:**

$$A = \begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix}, \quad \mathbf{v}_1 = \begin{bmatrix}1\\0\end{bmatrix},\quad \mathbf{v}_2 = \begin{bmatrix}0\\1\end{bmatrix}$$

$$A\mathbf{v}_1 = \begin{bmatrix}3\\0\end{bmatrix} = 3\mathbf{v}_1 \quad(\lambda_1=3), \qquad A\mathbf{v}_2 = \begin{bmatrix}0\\2\end{bmatrix} = 2\mathbf{v}_2 \quad(\lambda_2=2)$$

A non-eigenvector $\mathbf{u} = [1,1]^\top$ gets its direction changed:

$$A\mathbf{u} = \begin{bmatrix}3\\2\end{bmatrix} \neq \lambda\begin{bmatrix}1\\1\end{bmatrix} \quad \text{— rotated, not an eigenvector}$$

**How to find eigenvalues** — solve the characteristic equation:

$$\det(A - \lambda I) = 0$$

```python
A = np.array([[3.0, 1.0], [0.0, 2.0]])

eigenvalues, eigenvectors = np.linalg.eig(A)
# eigenvalues: [3., 2.]
# eigenvectors: columns — eigenvectors[:,0] corresponds to eigenvalue 3

# Verify: A @ v = λ * v
v   = eigenvectors[:, 0]
lam = eigenvalues[0]
np.allclose(A @ v, lam * v)   # True
```

### 6.2 Why Eigenvectors Matter

**1. They reveal natural axes of the transformation**

Eigenvectors expose the independent directions along which $A$ simply stretches or shrinks — no mixing, no rotation. In these axes the matrix behaves like a scalar.

**2. Repeated application becomes trivial**

$$A^k \mathbf{v} = \lambda^k \mathbf{v}$$

If $|\lambda| > 1$: repeated application explodes exponentially. If $|\lambda| < 1$: it decays to zero. This directly explains gradient explosion and vanishing in deep networks and RNNs.

**3. AI applications:**

| Application | How eigenvectors / eigenvalues are used |
|-------------|----------------------------------------|
| **PCA** | Eigenvectors of $\Sigma$ = directions of max variance; eigenvalues = variance in each direction |
| **Hessian analysis** | Large $\lambda_{\max}$ → sharp curvature → small learning rate required |
| **RNN stability** | $\|\lambda\| > 1$ → gradient explosion; $\|\lambda\| < 1$ → gradient vanishing |
| **Graph NNs** | Eigenvectors of the graph Laplacian encode graph frequency structure |
| **PageRank** | Dominant eigenvector of the web link matrix = page importance scores |

**4. Connection to Determinant:**

$$\det(A) = \prod_{i=1}^n \lambda_i$$

If any eigenvalue is zero → $\det(A) = 0$ → matrix is singular and not invertible.

### 6.3 Eigendecomposition

For a diagonalizable matrix $A$ with $n$ independent eigenvectors:

$$A = V \Lambda V^{-1}$$

| Term | Shape | Meaning |
|------|-------|---------|
| $V$ | $n \times n$ | Columns are eigenvectors — the natural axes of $A$ |
| $\Lambda$ | $n \times n$ diagonal | Diagonal entries are eigenvalues $\lambda_1, \ldots, \lambda_n$ |
| $V^{-1}$ | $n \times n$ | Change of basis back to original coordinates |

**Geometric reading:**

```
A x = V Λ V⁻¹ x

Step 1:  V⁻¹ x  — express x in eigenvector coordinates
Step 2:  Λ      — scale each component by its eigenvalue (no mixing)
Step 3:  V      — convert back to original coordinates
```

For **symmetric** $A$: eigenvectors are always orthogonal, so $V^{-1} = V^\top$:

$$A = V \Lambda V^\top \qquad \text{(spectral decomposition)}$$

```python
A = np.array([[4.0, 2.0], [2.0, 3.0]])
lam, V = np.linalg.eigh(A)   # eigh for symmetric — real, sorted eigenvalues

np.allclose(A, V @ np.diag(lam) @ V.T)   # True
np.allclose(np.linalg.det(A), np.prod(lam))  # det = product of eigenvalues ✓
```

---

## 7. Singular Value Decomposition (SVD)

### 7.1 Definition

SVD is the most important matrix decomposition in AI. It works for **any** matrix (not just square).

$$A = U \Sigma V^\top, \quad A \in \mathbb{R}^{m \times n}$$

| Term | Shape | Meaning |
|------|-------|---------|
| $U$ | $m \times m$ | Left singular vectors — orthogonal matrix |
| $\Sigma$ | $m \times n$ | Diagonal — singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$ |
| $V^\top$ | $n \times n$ | Right singular vectors — orthogonal matrix |

### 7.2 Geometric Meaning

Every linear transformation $A$ can be decomposed into three steps:

$$A\mathbf{x} = U\Sigma V^\top \mathbf{x}$$

```
Step 1:  V^T x  — rotate/reflect the input
Step 2:  Σ      — scale each dimension by σ_i
Step 3:  U      — rotate/reflect the output
```

The singular values $\sigma_i$ tell you how much $A$ stretches space along each direction. Large $\sigma_i$ = important direction; small $\sigma_i \approx 0$ = negligible direction.

```python
A = np.random.randn(5, 3)

U, S, Vt = np.linalg.svd(A, full_matrices=False)
# U: (5,3), S: (3,), Vt: (3,3)

# Reconstruct
A_reconstructed = U @ np.diag(S) @ Vt
np.allclose(A, A_reconstructed)   # True
```

### 7.3 Low-Rank Approximation

Keep only the top $k$ singular values — best possible rank-$k$ approximation of $A$:

$$A \approx A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top = U_k \Sigma_k V_k^\top$$

| Term | Meaning |
|------|---------|
| $\sigma_i$ | How important the $i$-th component is |
| $\mathbf{u}_i \mathbf{v}_i^\top$ | Rank-1 matrix — outer product of two vectors |
| $k \ll \min(m,n)$ | Keep only the most important components |

```python
U, S, Vt = np.linalg.svd(A, full_matrices=False)

k = 2
A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]   # rank-k approximation

# Compression ratio
print(f"Original: {A.shape}  →  Stored: {U[:,:k].size + k + Vt[:k].size}")
```

### 7.4 Why SVD Matters in AI

SVD is the most universally useful decomposition because it works for **any** matrix — not just square ones — and exposes the true geometric structure of any linear transformation.

**1. It reveals the effective rank of a matrix**

Singular values that are near zero mean those directions carry almost no information. The number of significantly nonzero singular values = the effective rank.

**2. It gives the best low-rank approximation (Eckart-Young theorem)**

No other rank-$k$ matrix is closer to $A$ than $A_k = U_k \Sigma_k V_k^\top$. This is optimal data compression.

**3. AI applications in depth:**

| Application | How SVD is used |
|-------------|----------------|
| **PCA** | SVD of centered data $\tilde{X} = U\Sigma V^\top$; columns of $V$ are principal components |
| **LoRA** | Pretrained weight matrices $W$ are approximately low-rank; learn only $\Delta W = AB$ (rank $r \ll d$) |
| **Recommendation systems** | Factorize user-item rating matrix $R \approx U\Sigma V^\top$ to predict missing ratings |
| **Noise removal** | Keep top-$k$ singular values; small $\sigma_i$ correspond to noise directions |
| **Condition number** | $\kappa(A) = \sigma_{\max}/\sigma_{\min}$; large → numerically unstable (gradients become unreliable) |
| **Pseudo-inverse** | $A^+ = V\Sigma^+ U^\top$ solves $A\mathbf{x}=\mathbf{b}$ in the least-squares sense even when $A$ is not invertible |

**4. SVD vs Eigendecomposition:**

| | Eigendecomposition | SVD |
|---|---|---|
| Applies to | Square matrices only | Any $m \times n$ matrix |
| Requires | Diagonalizable | Always exists |
| Output | Eigenvectors, eigenvalues | Left/right singular vectors, singular values |
| $\sigma_i$ vs $\lambda_i$ | Same for symmetric PSD | $\sigma_i = \sqrt{\lambda_i(A^\top A)}$ |

> **Key insight:** for symmetric PSD matrices (like covariance matrices), SVD and eigendecomposition coincide — $U = V$ and $\sigma_i = \lambda_i$. This is why PCA can be computed either way.

---

## 8. Vector Spaces & Subspaces

### 8.1 Linear Independence

A set of vectors $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ is **linearly independent** if the only way to combine them to get the zero vector is with all-zero coefficients:

$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0} \implies c_1 = c_2 = \cdots = c_k = 0$$

**Geometrically:** no vector in the set can be expressed as a combination of the others. Each vector adds a genuinely new direction.

**Linearly DEPENDENT example** (bad — one vector is redundant):

$$\mathbf{v}_1 = \begin{bmatrix}1\\0\end{bmatrix},\quad \mathbf{v}_2 = \begin{bmatrix}0\\1\end{bmatrix},\quad \mathbf{v}_3 = \begin{bmatrix}2\\3\end{bmatrix}$$

$\mathbf{v}_3 = 2\mathbf{v}_1 + 3\mathbf{v}_2$ — it adds no new direction. The three vectors only span a 2D plane, not 3D space.

**Linearly INDEPENDENT example** (good — all directions new):

$$\mathbf{v}_1 = \begin{bmatrix}1\\0\\0\end{bmatrix},\quad \mathbf{v}_2 = \begin{bmatrix}0\\1\\0\end{bmatrix},\quad \mathbf{v}_3 = \begin{bmatrix}0\\0\\1\end{bmatrix}$$

No vector is a combination of the others. Together they span all of $\mathbb{R}^3$.

**How to check:** put vectors as columns of $A$ — they are independent if and only if $\det(A) \neq 0$, or equivalently $\text{rank}(A) = k$.

```python
v1 = np.array([1., 0., 0.])
v2 = np.array([0., 1., 0.])
v3 = np.array([2., 3., 0.])   # dependent: v3 = 2v1 + 3v2

A = np.column_stack([v1, v2, v3])
np.linalg.matrix_rank(A)   # 2 — only 2 independent vectors, not 3
np.linalg.det(A)            # 0.0 — singular → dependent
```

**In AI:** redundant features (linearly dependent columns) make the model overparameterized and the matrix singular. PCA removes redundancy by finding independent directions.

---

### 8.2 Span, Basis & Dimension

**Span:** the set of all vectors reachable by linear combinations of $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$:

$$\text{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k) = \{c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k \mid c_i \in \mathbb{R}\}$$

**Basis:** a set of vectors that is (1) linearly independent AND (2) spans the whole space. It is the minimal complete description of a space.

$$\text{Standard basis of } \mathbb{R}^3: \quad \mathbf{e}_1 = \begin{bmatrix}1\\0\\0\end{bmatrix},\ \mathbf{e}_2 = \begin{bmatrix}0\\1\\0\end{bmatrix},\ \mathbf{e}_3 = \begin{bmatrix}0\\0\\1\end{bmatrix}$$

Every vector in $\mathbb{R}^3$ can be written uniquely as $c_1\mathbf{e}_1 + c_2\mathbf{e}_2 + c_3\mathbf{e}_3$.

**Dimension:** the number of vectors in any basis of the space.

$$\dim(\mathbb{R}^n) = n$$

| Space | Basis | Dimension |
|-------|-------|-----------|
| A line through the origin | 1 vector along the line | 1 |
| A plane through the origin | 2 non-parallel vectors in the plane | 2 |
| $\mathbb{R}^3$ | Any 3 independent vectors | 3 |
| Column space of a $5 \times 3$ rank-2 matrix | 2 independent column vectors | 2 |

> **Key insight:** dimension counts how many truly independent directions exist in a space. A $100 \times 100$ matrix with rank 5 only "lives in" a 5-dimensional subspace — despite having 100 rows and columns.

**Rank-Dimension relationship:**

$$\text{rank}(A) = \dim(\text{column space of } A)$$

```python
# Dimension of column space = rank
A = np.array([[1., 2., 3.],
              [4., 5., 6.],
              [7., 8., 9.]])   # rank-deficient: row3 = row1 + row2

np.linalg.matrix_rank(A)   # 2 — only 2D column space, not 3D
```

---

### 8.3 Rank & Null Space

**Rank** of $A$ = number of linearly independent columns = dimension of the column space.

$$\text{rank}(A) = r \leq \min(m, n)$$

| rank$(A)$ | Meaning |
|-----------|---------|
| $r = n$ (full column rank) | Columns are independent — $A\mathbf{x}=\mathbf{b}$ has at most one solution |
| $r = m$ (full row rank) | Every output is reachable — $A\mathbf{x}=\mathbf{b}$ always has a solution |
| $r < \min(m,n)$ | Rank-deficient — $A$ destroys information (collapses some directions to zero) |

**Null Space (Kernel):** the set of all input vectors that $A$ maps to the zero vector.

$$\text{null}(A) = \{\mathbf{x} \mid A\mathbf{x} = \mathbf{0}\}$$

**Geometric meaning:** the null space is the set of directions that $A$ completely squashes to zero — they carry no information through $A$.

**Rank-Nullity Theorem:**

$$\text{rank}(A) + \dim(\text{null}(A)) = n \quad \text{(number of columns)}$$

The more dimensions $A$ collapses (large null space), the fewer independent directions survive (low rank).

| rank$(A)$ | $\dim(\text{null}(A))$ | Meaning |
|-----------|------------------------|---------|
| $n$ (full) | $0$ | Trivial null space — only $\mathbf{x}=\mathbf{0}$ maps to $\mathbf{0}$ — $A$ is invertible |
| $n-1$ | $1$ | One direction collapses — $A$ is singular |
| $0$ | $n$ | Everything maps to $\mathbf{0}$ — $A$ is the zero matrix |

**Concrete example:**

$$A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$$

Column 2 = $2 \times$ Column 1 → rank = 1. The null space is all $\mathbf{x}$ satisfying $x_1 + 2x_2 = 0$:

$$\text{null}(A) = \left\{ t \begin{bmatrix} -2 \\ 1 \end{bmatrix} \mid t \in \mathbb{R} \right\} \quad \text{(a line through the origin)}$$

```python
A = np.array([[1., 2.], [2., 4.]])

np.linalg.matrix_rank(A)   # 1

# Null space via SVD: right singular vectors with σ ≈ 0
U, S, Vt = np.linalg.svd(A)
null_space = Vt[S < 1e-10]   # rows of Vt where singular value ≈ 0
# [[-0.894,  0.447]] ≈ direction [-2, 1] normalized
```

**In AI:** if a weight matrix $W$ has a large null space, many input directions are completely ignored — the layer has low effective rank. LoRA exploits this: large pretrained weight matrices tend to be approximately low-rank, so you only need to learn a small rank update $\Delta W = AB$.

---

## 9. Orthogonality

### 9.1 Orthogonal Vectors & Matrices

Two vectors are **orthogonal** if their dot product is zero:

$$\mathbf{a} \perp \mathbf{b} \iff \mathbf{a}^\top \mathbf{b} = 0$$

An **orthonormal** set has unit-length mutually orthogonal vectors:

$$\mathbf{q}_i^\top \mathbf{q}_j = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}$$

An **orthogonal matrix** $Q$ has orthonormal columns: $Q^\top Q = I$, so $Q^{-1} = Q^\top$.

**Why orthogonal matrices matter in AI:**
- They preserve the L2 norm: $\|Q\mathbf{x}\|_2 = \|\mathbf{x}\|_2$
- They preserve dot products: $(Q\mathbf{a})^\top(Q\mathbf{b}) = \mathbf{a}^\top\mathbf{b}$
- Gradients don't explode or vanish through them — used in RNNs and normalizing flows

### 9.2 Projections

The **projection** of $\mathbf{b}$ onto $\mathbf{a}$ is the component of $\mathbf{b}$ in the direction of $\mathbf{a}$:

$$\text{proj}_{\mathbf{a}} \mathbf{b} = \frac{\mathbf{a}^\top \mathbf{b}}{\mathbf{a}^\top \mathbf{a}} \mathbf{a}$$

The **projection matrix** onto the column space of $A$:

$$P = A(A^\top A)^{-1} A^\top, \qquad P^2 = P \quad \text{(idempotent)}$$

| Term | Meaning |
|------|---------|
| $P\mathbf{b}$ | Closest point in column space of $A$ to $\mathbf{b}$ |
| $\mathbf{b} - P\mathbf{b}$ | Residual — perpendicular to column space |
| $P^2 = P$ | Projecting twice is the same as projecting once |

**In AI:** attention computes projections; least squares regression minimizes $\|A\mathbf{x} - \mathbf{b}\|_2^2$ via projection.

---

## 10. Positive Definite Matrices

A symmetric matrix $A$ is **positive definite (PD)** if:

$$\mathbf{x}^\top A \mathbf{x} > 0 \quad \forall \mathbf{x} \neq \mathbf{0}$$

Equivalently: all eigenvalues are strictly positive.

| Type | Condition | Eigenvalues |
|------|-----------|-------------|
| Positive definite (PD) | $\mathbf{x}^\top A \mathbf{x} > 0$ | All $\lambda > 0$ |
| Positive semi-definite (PSD) | $\mathbf{x}^\top A \mathbf{x} \geq 0$ | All $\lambda \geq 0$ |
| Indefinite | Neither | Mixed signs |

**Why AI cares:**
- The **Hessian** of the loss being PD means the loss is convex — guaranteed unique minimum
- **Covariance matrices** are always PSD — eigenvalues represent variance along each direction
- **Kernel matrices** in SVMs must be PSD
- Adam optimizer uses a diagonal approximation of the PD Hessian

```python
A = np.array([[4.0, 2.0], [2.0, 3.0]])

eigenvalues = np.linalg.eigvalsh(A)
is_pd  = np.all(eigenvalues > 0)    # True if positive definite
is_psd = np.all(eigenvalues >= 0)   # True if positive semi-definite
```

---

## 11. Gradients & Jacobians

### 11.1 Gradient as a Vector

For a scalar function $f: \mathbb{R}^n \to \mathbb{R}$, the gradient is the vector of partial derivatives:

$$\nabla_{\mathbf{x}} f = \begin{bmatrix} \partial f / \partial x_1 \\ \partial f / \partial x_2 \\ \vdots \\ \partial f / \partial x_n \end{bmatrix} \in \mathbb{R}^n$$

The gradient points in the direction of **steepest increase**. Gradient descent steps opposite to it:

$$\mathbf{x} \leftarrow \mathbf{x} - \alpha \nabla_\mathbf{x} f(\mathbf{x})$$

**Example — L2 loss:**

$$f(\mathbf{x}) = \|\mathbf{x}\|_2^2 = x_1^2 + x_2^2 + \cdots + x_n^2$$

$$\nabla_\mathbf{x} f = 2\mathbf{x}$$

**Common gradient rules:**

| Function | Gradient |
|----------|----------|
| $f(\mathbf{x}) = \mathbf{a}^\top \mathbf{x}$ | $\nabla f = \mathbf{a}$ |
| $f(\mathbf{x}) = \mathbf{x}^\top A \mathbf{x}$ | $\nabla f = (A + A^\top)\mathbf{x} = 2A\mathbf{x}$ (if $A$ symmetric) |
| $f(\mathbf{x}) = \|\mathbf{x}\|_2^2$ | $\nabla f = 2\mathbf{x}$ |
| $f(W) = \mathbf{a}^\top W \mathbf{b}$ | $\nabla_W f = \mathbf{a}\mathbf{b}^\top$ |

### 11.2 Jacobian Matrix

For a vector-valued function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian contains all partial derivatives:

$$J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix} \partial f_1/\partial x_1 & \cdots & \partial f_1/\partial x_n \\ \vdots & \ddots & \vdots \\ \partial f_m/\partial x_1 & \cdots & \partial f_m/\partial x_n \end{bmatrix} \in \mathbb{R}^{m \times n}$$

| Term | Meaning |
|------|---------|
| $J_{ij}$ | How output $i$ changes when input $j$ changes |
| $J \in \mathbb{R}^{m \times n}$ | $m$ outputs, $n$ inputs |
| $\det(J)$ | Local volume scaling factor (used in normalizing flows) |

**In AI:** backpropagation computes Jacobian-vector products efficiently using the chain rule:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = J^\top \frac{\partial \mathcal{L}}{\partial \mathbf{f}}$$

### 11.3 Hessian Matrix

For $f: \mathbb{R}^n \to \mathbb{R}$, the Hessian is the matrix of second derivatives:

$$H = \nabla^2 f = \begin{bmatrix} \partial^2 f/\partial x_1^2 & \partial^2 f/\partial x_1 \partial x_2 & \cdots \\ \partial^2 f/\partial x_2 \partial x_1 & \partial^2 f/\partial x_2^2 & \cdots \\ \vdots & & \ddots \end{bmatrix} \in \mathbb{R}^{n \times n}$$

| Hessian property | Meaning for the loss landscape |
|------------------|-------------------------------|
| $H \succ 0$ (PD) | Local minimum — convex bowl |
| $H$ has negative eigenvalue | Saddle point — flat or descending in some direction |
| Large $\lambda_{\max}(H)$ | Sharp curvature — small learning rate required |
| Small $\lambda_{\max}(H)$ | Flat landscape — can use larger learning rate |

**In AI:** the ratio $\lambda_{\max} / \lambda_{\min}$ of the Hessian is the **condition number** — large condition number means training is slow and unstable (requires adaptive optimizers like Adam).

---

## 12. Matrix Derivatives

Matrix derivatives are the engine of backpropagation. Every gradient update in a neural network is a matrix derivative computed through the chain rule.

### 12.1 Layout Convention

There are two layout conventions. This document uses the **denominator layout** (also called Hessian layout), which is most common in ML:

| Derivative | Numerator shape | Denominator shape | Result shape |
|------------|----------------|-------------------|-------------|
| $\partial y / \partial x$ | scalar | scalar | scalar |
| $\partial y / \partial \mathbf{x}$ | scalar | $(n,)$ | $(n,)$ column vector |
| $\partial \mathbf{y} / \partial x$ | $(m,)$ | scalar | $(m,)$ column vector |
| $\partial \mathbf{y} / \partial \mathbf{x}$ | $(m,)$ | $(n,)$ | $(n \times m)$ Jacobian |
| $\partial y / \partial X$ | scalar | $(m \times n)$ | $(m \times n)$ matrix |

> **Rule of thumb:** the result has the same shape as the denominator (what you differentiate with respect to).

---

### 12.2 Scalar by Vector — $\partial y / \partial \mathbf{x}$

The most common case: loss function output w.r.t. input vector.

**Result shape:** same as $\mathbf{x}$ — a column vector.

---

**Example 1 — Linear function: $y = \mathbf{a}^\top \mathbf{x}$**

$$y = \mathbf{a}^\top \mathbf{x} = a_1 x_1 + a_2 x_2 + \cdots + a_n x_n$$

$$\frac{\partial y}{\partial x_i} = a_i \quad \Rightarrow \quad \frac{\partial y}{\partial \mathbf{x}} = \mathbf{a}$$

**Concrete numbers** ($n=3$):

$$\mathbf{a} = \begin{bmatrix}2\\3\\-1\end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix}1\\0\\4\end{bmatrix}, \quad y = 2(1)+3(0)+(-1)(4) = -2$$

$$\frac{\partial y}{\partial \mathbf{x}} = \begin{bmatrix}2\\3\\-1\end{bmatrix} = \mathbf{a} \qquad \text{(constant — doesn't depend on } \mathbf{x}\text{)}$$

---

**Example 2 — L2 norm squared: $y = \|\mathbf{x}\|_2^2 = \mathbf{x}^\top\mathbf{x}$**

$$y = x_1^2 + x_2^2 + \cdots + x_n^2$$

$$\frac{\partial y}{\partial x_i} = 2x_i \quad \Rightarrow \quad \frac{\partial y}{\partial \mathbf{x}} = 2\mathbf{x}$$

**Concrete numbers** ($\mathbf{x} = [1, 2, 3]^\top$):

$$y = 1 + 4 + 9 = 14, \qquad \frac{\partial y}{\partial \mathbf{x}} = \begin{bmatrix}2\\4\\6\end{bmatrix}$$

```python
x = np.array([1., 2., 3.])
y = x @ x                    # 14.0
grad = 2 * x                 # [2., 4., 6.]
```

---

**Example 3 — Quadratic form: $y = \mathbf{x}^\top A \mathbf{x}$**

$$y = \sum_i \sum_j x_i A_{ij} x_j$$

$$\frac{\partial y}{\partial \mathbf{x}} = (A + A^\top)\mathbf{x} = 2A\mathbf{x} \quad \text{(if } A \text{ symmetric)}$$

**Concrete numbers** ($A$ symmetric, $2 \times 2$):

$$A = \begin{bmatrix}2 & 1\\1 & 3\end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix}1\\2\end{bmatrix}$$

$$y = \begin{bmatrix}1&2\end{bmatrix}\begin{bmatrix}2&1\\1&3\end{bmatrix}\begin{bmatrix}1\\2\end{bmatrix} = \begin{bmatrix}1&2\end{bmatrix}\begin{bmatrix}4\\7\end{bmatrix} = 4+14 = 18$$

$$\frac{\partial y}{\partial \mathbf{x}} = 2A\mathbf{x} = 2\begin{bmatrix}2&1\\1&3\end{bmatrix}\begin{bmatrix}1\\2\end{bmatrix} = 2\begin{bmatrix}4\\7\end{bmatrix} = \begin{bmatrix}8\\14\end{bmatrix}$$

---

### 12.3 Scalar by Matrix — $\partial y / \partial W$

Used every time we update neural network weights.

**Result shape:** same as $W$ — an $(m \times n)$ matrix.

**Key rule:** $({\partial y}/{\partial W})_{ij}$ = how much $y$ changes when $W_{ij}$ changes, holding all other entries fixed.

---

**Example 4 — Linear layer loss: $y = \mathbf{a}^\top W \mathbf{b}$**

$$y = \sum_i \sum_j a_i W_{ij} b_j$$

$$\frac{\partial y}{\partial W_{ij}} = a_i b_j \quad \Rightarrow \quad \frac{\partial y}{\partial W} = \mathbf{a}\mathbf{b}^\top$$

**Concrete numbers** ($m=2, n=3$):

$$\mathbf{a} = \begin{bmatrix}2\\3\end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix}1\\0\\-1\end{bmatrix}$$

$$\frac{\partial y}{\partial W} = \mathbf{a}\mathbf{b}^\top = \begin{bmatrix}2\\3\end{bmatrix}\begin{bmatrix}1&0&-1\end{bmatrix} = \begin{bmatrix}2&0&-2\\3&0&-3\end{bmatrix}$$

Each entry $(i,j)$ tells how changing $W_{ij}$ affects $y$.

---

**Example 5 — MSE loss w.r.t. weight matrix**

$$\mathcal{L} = \|\mathbf{y} - W\mathbf{x}\|_2^2 = (\mathbf{y} - W\mathbf{x})^\top(\mathbf{y} - W\mathbf{x})$$

Let $\mathbf{e} = \mathbf{y} - W\mathbf{x}$ (residual vector). Then:

$$\frac{\partial \mathcal{L}}{\partial W} = -2\,\mathbf{e}\,\mathbf{x}^\top$$

**Concrete numbers** ($m=2, n=2$):

$$W = \begin{bmatrix}1&0\\0&1\end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix}2\\1\end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix}3\\3\end{bmatrix}$$

$$W\mathbf{x} = \begin{bmatrix}2\\1\end{bmatrix}, \quad \mathbf{e} = \mathbf{y} - W\mathbf{x} = \begin{bmatrix}1\\2\end{bmatrix}$$

$$\frac{\partial \mathcal{L}}{\partial W} = -2\begin{bmatrix}1\\2\end{bmatrix}\begin{bmatrix}2&1\end{bmatrix} = -2\begin{bmatrix}2&1\\4&2\end{bmatrix} = \begin{bmatrix}-4&-2\\-8&-4\end{bmatrix}$$

The negative sign means: step in the opposite direction to reduce loss.

```python
W = np.eye(2)
x = np.array([2., 1.])
y = np.array([3., 3.])

e    = y - W @ x                     # residual: [1., 2.]
grad = -2 * np.outer(e, x)           # [[-4,-2],[-8,-4]]

W_new = W - 0.1 * grad               # gradient descent step
```

---

### 12.4 Vector by Vector — Jacobian $\partial \mathbf{y} / \partial \mathbf{x}$

When both input and output are vectors, the derivative is a **Jacobian matrix**.

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = J \in \mathbb{R}^{n \times m}, \qquad J_{ji} = \frac{\partial y_j}{\partial x_i}$$

---

**Example 6 — Linear layer: $\mathbf{y} = W\mathbf{x}$**

$$y_i = \sum_j W_{ij} x_j \quad \Rightarrow \quad \frac{\partial y_i}{\partial x_j} = W_{ij}$$

$$J = \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = W$$

The Jacobian of a linear layer is simply the weight matrix $W$ itself.

---

**Example 7 — ReLU: $\mathbf{y} = \text{ReLU}(\mathbf{x})$, i.e. $y_i = \max(0, x_i)$**

ReLU operates element-wise so the Jacobian is **diagonal**:

$$\frac{\partial y_i}{\partial x_j} = \begin{cases} 1 & \text{if } i = j \text{ and } x_i > 0 \\ 0 & \text{otherwise} \end{cases}$$

$$J = \text{diag}(\mathbf{1}[\mathbf{x} > 0]) = \begin{bmatrix} \mathbf{1}[x_1>0] & & \\ & \ddots & \\ & & \mathbf{1}[x_n>0] \end{bmatrix}$$

**Concrete numbers** ($\mathbf{x} = [-1, 2, -0.5, 3]^\top$):

$$J = \begin{bmatrix}0&0&0&0\\0&1&0&0\\0&0&0&0\\0&0&0&1\end{bmatrix}$$

Dead neurons ($x_i \leq 0$) contribute zero gradient — they do not participate in learning.

---

**Example 8 — Softmax: $\mathbf{p} = \text{softmax}(\mathbf{z})$**

$$p_i = \frac{e^{z_i}}{\sum_k e^{z_k}}$$

$$\frac{\partial p_i}{\partial z_j} = \begin{cases} p_i(1 - p_i) & i = j \\ -p_i p_j & i \neq j \end{cases}$$

In matrix form:

$$J = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$$

**Concrete numbers** ($\mathbf{z} = [2, 1, 0]^\top$):

$$\mathbf{p} = \text{softmax}(\mathbf{z}) \approx [0.665, 0.245, 0.090]$$

$$J = \begin{bmatrix}0.665&0&0\\0&0.245&0\\0&0&0.090\end{bmatrix} - \begin{bmatrix}0.665\\0.245\\0.090\end{bmatrix}\begin{bmatrix}0.665&0.245&0.090\end{bmatrix}$$

$$= \begin{bmatrix}0.665&0&0\\0&0.245&0\\0&0&0.090\end{bmatrix} - \begin{bmatrix}0.442&0.163&0.060\\0.163&0.060&0.022\\0.060&0.022&0.008\end{bmatrix}$$

$$= \begin{bmatrix}0.223&-0.163&-0.060\\-0.163&0.185&-0.022\\-0.060&-0.022&0.082\end{bmatrix}$$

```python
z = np.array([2., 1., 0.])
p = np.exp(z) / np.exp(z).sum()     # [0.665, 0.245, 0.090]

J = np.diag(p) - np.outer(p, p)    # (3,3) Jacobian of softmax
```

---

### 12.5 Chain Rule in Matrix Form

Backpropagation is the chain rule applied repeatedly through layers.

For $\mathcal{L} = f(g(\mathbf{x}))$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \underbrace{\frac{\partial g}{\partial \mathbf{x}}}_{J_g^\top} \cdot \underbrace{\frac{\partial \mathcal{L}}{\partial g}}_{\text{upstream gradient}}$$

---

**Example 9 — One linear layer + MSE loss**

Forward pass:

$$\mathbf{h} = W\mathbf{x}, \quad \mathcal{L} = \|\mathbf{y} - \mathbf{h}\|_2^2$$

**Step 1 — $\partial \mathcal{L} / \partial \mathbf{h}$** (loss w.r.t. layer output):

$$\mathcal{L} = \sum_i (y_i - h_i)^2 \quad \Rightarrow \quad \frac{\partial \mathcal{L}}{\partial \mathbf{h}} = -2(\mathbf{y} - \mathbf{h}) = -2\mathbf{e}$$

**Step 2 — $\partial \mathcal{L} / \partial W$** (chain rule):

$$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}} \cdot \mathbf{x}^\top = -2\mathbf{e}\mathbf{x}^\top$$

**Step 3 — $\partial \mathcal{L} / \partial \mathbf{x}$** (pass gradient to previous layer):

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = W^\top \frac{\partial \mathcal{L}}{\partial \mathbf{h}} = -2W^\top\mathbf{e}$$

**Concrete numbers:**

$$W = \begin{bmatrix}1&2\\3&4\end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix}1\\1\end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix}5\\8\end{bmatrix}$$

```python
W = np.array([[1., 2.], [3., 4.]])
x = np.array([1., 1.])
y = np.array([5., 8.])

# Forward
h = W @ x                           # [3., 7.]
L = np.sum((y - h)**2)              # (5-3)² + (8-7)² = 5.0

# Backward
e         = y - h                   # [2., 1.]
dL_dh     = -2 * e                  # [-4., -2.]
dL_dW     = np.outer(dL_dh, x)     # [[-4,-4],[-2,-2]]
dL_dx     = W.T @ dL_dh             # [-4*1+(-2)*3, -4*2+(-2)*4] = [-10, -16]

# Gradient descent on W
W_new = W - 0.01 * dL_dW
```

---

**Example 10 — Linear layer + ReLU + MSE (two-step backprop)**

$$\mathbf{z} = W\mathbf{x}, \quad \mathbf{h} = \text{ReLU}(\mathbf{z}), \quad \mathcal{L} = \|\mathbf{y} - \mathbf{h}\|_2^2$$

```python
W = np.array([[1., -1.], [2., 3.]])
x = np.array([1., 2.])
y = np.array([1., 5.])

# Forward
z = W @ x                           # [-1., 8.]
h = np.maximum(0, z)                # [ 0., 8.]  ← ReLU
L = np.sum((y - h)**2)              # (1-0)² + (5-8)² = 10.0

# Backward
e         = y - h                   # [1., -3.]
dL_dh     = -2 * e                  # [-2., 6.]

# ReLU Jacobian: pass gradient only where z > 0
relu_mask = (z > 0).astype(float)  # [0., 1.]
dL_dz     = dL_dh * relu_mask      # [-2.*0, 6.*1] = [0., 6.]

# Gradient w.r.t. W
dL_dW     = np.outer(dL_dz, x)     # [[0,0],[6,12]]

# Gradient w.r.t. x (pass to previous layer)
dL_dx     = W.T @ dL_dz            # [0*1+6*2, 0*(-1)+6*3] = [12., 18.]
```

---

### 12.6 Common Matrix Derivative Reference

| Function $f$ | Derivative $\partial f / \partial \mathbf{x}$ or $\partial f / \partial W$ |
|---|---|
| $\mathbf{a}^\top \mathbf{x}$ | $\mathbf{a}$ |
| $\mathbf{x}^\top \mathbf{x}$ | $2\mathbf{x}$ |
| $\mathbf{x}^\top A \mathbf{x}$ | $2A\mathbf{x}$ (symmetric $A$) |
| $\|\mathbf{y} - W\mathbf{x}\|_2^2$ w.r.t. $\mathbf{x}$ | $-2W^\top(\mathbf{y} - W\mathbf{x})$ |
| $\|\mathbf{y} - W\mathbf{x}\|_2^2$ w.r.t. $W$ | $-2(\mathbf{y} - W\mathbf{x})\mathbf{x}^\top$ |
| $\mathbf{a}^\top W \mathbf{b}$ w.r.t. $W$ | $\mathbf{a}\mathbf{b}^\top$ |
| $\text{tr}(A^\top W)$ w.r.t. $W$ | $A$ |
| $\log\det(W)$ w.r.t. $W$ | $W^{-\top}$ |
| $\text{ReLU}(\mathbf{x})$ | $\text{diag}(\mathbf{1}[\mathbf{x}>0])$ |
| $\text{softmax}(\mathbf{z})$ | $\text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$ |
| $\sigma(x)$ (sigmoid) | $\sigma(x)(1-\sigma(x))$ |

---

## 13. AI Applications

### 13.1 Neural Network Forward Pass

A fully connected layer is a matrix multiplication followed by a nonlinearity:

$$\mathbf{h}^{(l)} = \sigma(W^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)})$$

For a batch of $N$ samples:

$$H^{(l)} = \sigma(H^{(l-1)} W^{(l)\top} + \mathbf{b}^{(l)\top}), \quad H \in \mathbb{R}^{N \times d}$$

| Matrix | Shape | Role |
|--------|-------|------|
| $W^{(l)}$ | $d_\text{out} \times d_\text{in}$ | Learnable weights — linear transformation |
| $\mathbf{b}^{(l)}$ | $d_\text{out}$ | Bias — shifts the output |
| $\mathbf{h}^{(l-1)}$ | $d_\text{in}$ | Input from previous layer |
| $\mathbf{h}^{(l)}$ | $d_\text{out}$ | Output to next layer |

```python
W = np.random.randn(256, 512)   # (out, in)
b = np.zeros(256)
x = np.random.randn(512)

h = np.maximum(0, W @ x + b)   # ReLU activation
```

### 13.2 Attention Mechanism

Scaled dot-product attention computes similarity between queries and keys, then uses it to weight values:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

| Matrix | Shape | Role |
|--------|-------|------|
| $Q$ | $n \times d_k$ | Queries — what we are looking for |
| $K$ | $m \times d_k$ | Keys — what each token offers |
| $V$ | $m \times d_v$ | Values — what each token contains |
| $QK^\top$ | $n \times m$ | Similarity scores — dot product between every query and key |
| $\sqrt{d_k}$ | scalar | Scaling — prevents dot products from growing large in high dimensions |

$$QK^\top \in \mathbb{R}^{n \times m}: \quad (QK^\top)_{ij} = \mathbf{q}_i^\top \mathbf{k}_j \quad \text{(similarity of query } i \text{ to key } j\text{)}$$

```python
import torch
import torch.nn.functional as F

d_k = 64
Q = torch.randn(10, d_k)    # 10 queries
K = torch.randn(20, d_k)    # 20 keys
V = torch.randn(20, 128)    # 20 values, dim 128

scores  = Q @ K.T / d_k**0.5     # (10, 20) similarity matrix
weights = F.softmax(scores, dim=-1)  # (10, 20) attention weights
output  = weights @ V             # (10, 128) weighted sum of values
```

### 13.3 PCA

Principal Component Analysis finds the directions of maximum variance in data using SVD.

**Steps:**

$$\text{1. Center: } \tilde{X} = X - \bar{X}$$
$$\text{2. SVD: } \tilde{X} = U\Sigma V^\top$$
$$\text{3. Project: } Z = \tilde{X} V_k = U_k \Sigma_k \quad \text{(top-}k\text{ components)}$$

| Term | Meaning |
|------|---------|
| $V$ columns | **Principal components** — directions of maximum variance |
| $\Sigma$ diagonal | **Standard deviations** along each principal component |
| $\sigma_i^2$ | Variance explained by component $i$ |
| $k$ | Number of dimensions to keep |

```python
X = np.random.randn(100, 50)     # 100 samples, 50 features

X_centered = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

k = 10
Z = X_centered @ Vt[:k].T       # (100, 10) — projected to 10 dims

variance_explained = S[:k]**2 / (S**2).sum()
print(f"Variance explained by top {k} PCs: {variance_explained.sum():.2%}")
```

**In AI:**
- Reduce input dimensionality before training
- Visualize high-dimensional embeddings (word vectors, latent spaces)
- Initialize weights or analyze learned representations
