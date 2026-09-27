# Linear Algebra

The study of vectors, matrices, and linear maps. It is the language of systems of equations, geometry,
data analysis, dynamical systems, and physics — and the mathematical backbone of modern AI, where neural
networks, attention, PCA, SVD, and optimization all reduce to matrix and vector operations.

## Table of Contents

1. [Vectors](#1-vectors)
   - 1.1 [Definition & Notation](#11-definition--notation)
   - 1.2 [Vector Operations](#12-vector-operations)
   - 1.3 [Dot Product](#13-dot-product)
   - 1.4 [Norms](#14-norms)
   - 1.5 [Cross Product & Triple Product](#15-cross-product--triple-product)
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
   - 4.3 [Gaussian Elimination & Row Reduction](#43-gaussian-elimination--row-reduction)
   - 4.4 [LU Decomposition](#44-lu-decomposition)
   - 4.5 [Elementary Matrices & Row Equivalence](#45-elementary-matrices--row-equivalence)
5. [Determinant](#5-determinant)
   - 5.1 [Definition & Geometric Meaning](#51-definition--geometric-meaning)
   - 5.2 [Properties](#52-properties)
   - 5.3 [Cofactor (Laplace) Expansion](#53-cofactor-laplace-expansion)
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
   - 8.4 [The Four Fundamental Subspaces](#84-the-four-fundamental-subspaces)
9. [Orthogonality](#9-orthogonality)
   - 9.1 [Orthogonal Vectors & Matrices](#91-orthogonal-vectors--matrices)
   - 9.2 [Projections](#92-projections)
   - 9.3 [Gram–Schmidt Orthogonalization](#93-gramschmidt-orthogonalization)
   - 9.4 [QR Factorization](#94-qr-factorization)
10. [Positive Definite Matrices](#10-positive-definite-matrices)
11. [Gradients & Jacobians](#11-gradients--jacobians)
    - 11.1 [Gradient as a Vector](#111-gradient-as-a-vector)
    - 11.2 [Jacobian Matrix](#112-jacobian-matrix)
    - 11.3 [Hessian Matrix](#113-hessian-matrix)
12. [Matrix Derivatives](#12-matrix-derivatives)
    - 12.1 [Layout Convention](#121-layout-convention)
    - 12.2 [Scalar by Vector](#122-scalar-by-vector)
    - 12.3 [Scalar by Matrix](#123-scalar-by-matrix)
    - 12.4 [Vector by Vector — Jacobian](#124-vector-by-vector--jacobian)
    - 12.5 [Chain Rule in Matrix Form](#125-chain-rule-in-matrix-form)
    - 12.6 [Common Reference Table](#126-common-matrix-derivative-reference)
13. [Least Squares](#13-least-squares)
    - 13.1 [The Overdetermined Problem](#131-the-overdetermined-problem)
    - 13.2 [Normal Equations](#132-normal-equations)
    - 13.3 [Solving via QR](#133-solving-via-qr)
    - 13.4 [Polynomial Fitting & the Design Matrix](#134-polynomial-fitting--the-design-matrix)
    - 13.5 [Regularization (Ridge / Tikhonov)](#135-regularization-ridge--tikhonov)
    - 13.6 [Equality-Constrained Least Squares (KKT)](#136-equality-constrained-least-squares-kkt)
    - 13.7 [Reference Table](#137-reference-table)
14. [Linear Dynamical Systems](#14-linear-dynamical-systems)
    - 14.1 [Discrete Systems & Matrix Powers](#141-discrete-systems--matrix-powers)
    - 14.2 [Stability & Steady States](#142-stability--steady-states)
    - 14.3 [Markov Chains](#143-markov-chains)
    - 14.4 [Continuous Systems & the Matrix Exponential](#144-continuous-systems--the-matrix-exponential)
    - 14.5 [The Graph Laplacian](#145-the-graph-laplacian)
15. [AI Applications](#15-ai-applications)
    - 15.1 [Neural Network Forward Pass](#151-neural-network-forward-pass)
    - 15.2 [Attention Mechanism](#152-attention-mechanism)
    - 15.3 [PCA](#153-pca)
    - 15.4 [K-Means Clustering](#154-k-means-clustering)
    - 15.5 [K-Nearest Neighbors (KNN)](#155-k-nearest-neighbors-knn)

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



### 1.2 Vector Operations

**Addition and scalar multiplication:**

$$\mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \end{bmatrix}, \qquad c\mathbf{a} = \begin{bmatrix} ca_1 \\ ca_2 \end{bmatrix}$$

**Geometric meaning:** addition moves along one vector then another; scalar multiplication stretches or shrinks.



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



### 1.4 Norms

A norm measures the **length (magnitude)** of a vector.

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2} \qquad (\text{L2 norm — Euclidean length})$$

$$\|\mathbf{x}\|_1 = \sum_{i=1}^n |x_i| \qquad (\text{L1 norm — Manhattan distance})$$

$$\|\mathbf{x}\|_\infty = \max_i |x_i| \qquad (\text{L}\infty\text{ norm — largest element})$$

| Norm | Used in AI for |
|------|----------------|
| L2 ($\|\cdot\|_2$) | Weight decay, gradient clipping, Euclidean distance |
| L1 ($\|\cdot\|_1$) | Sparse regularization (Lasso) |
| $L^\infty$ | Adversarial robustness bounds |



### 1.5 Cross Product & Triple Product

The **cross product** is defined only in $\mathbb{R}^3$. It takes two vectors and returns a
**third vector perpendicular to both**:

$$\mathbf{a} \times \mathbf{b} = \begin{bmatrix} a_2 b_3 - a_3 b_2 \\ a_3 b_1 - a_1 b_3 \\ a_1 b_2 - a_2 b_1 \end{bmatrix} = \det\begin{bmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ a_1 & a_2 & a_3 \\ b_1 & b_2 & b_3 \end{bmatrix}$$

| Property | Meaning |
|----------|---------|
| $\|\mathbf{a} \times \mathbf{b}\| = \|\mathbf{a}\|\|\mathbf{b}\|\sin\theta$ | Magnitude = area of the parallelogram spanned by $\mathbf{a},\mathbf{b}$ |
| $\mathbf{a} \times \mathbf{b} \perp \mathbf{a}$ and $\perp \mathbf{b}$ | Direction given by the right-hand rule |
| $\mathbf{a} \times \mathbf{b} = -(\mathbf{b} \times \mathbf{a})$ | Anticommutative |
| $\mathbf{a} \times \mathbf{b} = \mathbf{0}$ | $\mathbf{a}, \mathbf{b}$ are parallel (linearly dependent) |

**Scalar triple product** — combines a dot and a cross product into one scalar:

$$\mathbf{a} \cdot (\mathbf{b} \times \mathbf{c}) = \det\begin{bmatrix} a_1 & a_2 & a_3 \\ b_1 & b_2 & b_3 \\ c_1 & c_2 & c_3 \end{bmatrix}$$

| Value | Geometric meaning |
|-------|-------------------|
| $\lvert \mathbf{a} \cdot (\mathbf{b} \times \mathbf{c}) \rvert$ | Volume of the parallelepiped spanned by $\mathbf{a}, \mathbf{b}, \mathbf{c}$ |
| $\mathbf{a} \cdot (\mathbf{b} \times \mathbf{c}) = 0$ | The three vectors are **coplanar** — linearly dependent (§8.1) |
| Sign | Orientation (handedness) of the ordered triple |

**Concrete example:**

$$\mathbf{a} = \begin{bmatrix}1\\0\\0\end{bmatrix},\ \mathbf{b} = \begin{bmatrix}0\\1\\0\end{bmatrix},\ \mathbf{c} = \begin{bmatrix}0\\0\\1\end{bmatrix} \ \Rightarrow\ \mathbf{b}\times\mathbf{c} = \begin{bmatrix}1\\0\\0\end{bmatrix},\quad \mathbf{a}\cdot(\mathbf{b}\times\mathbf{c}) = 1$$

Unit cube → volume 1. Replacing $\mathbf{c}$ by $\mathbf{a}+\mathbf{b}$ gives triple product $0$ (the three vectors now lie in a plane).

**Uses:** surface normals in graphics, torque and angular momentum in physics ($\boldsymbol{\tau} = \mathbf{r}\times\mathbf{F}$),
and a cheap linear-independence / coplanarity test for three vectors in $\mathbb{R}^3$.

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



### 2.3 Matrix Multiplication

$$C = AB \in \mathbb{R}^{m \times p}, \quad A \in \mathbb{R}^{m \times n},\quad B \in \mathbb{R}^{n \times p}$$

$$c_{ij} = \sum_{k=1}^n a_{ik} b_{kj} = \text{(row } i \text{ of } A) \cdot \text{(column } j \text{ of } B)$$

**Shape rule:** inner dimensions must match — $(m \times \mathbf{n})(\mathbf{n} \times p) = (m \times p)$





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

**Proofs of the four properties**

Throughout, $A \in \mathbb{R}^{n \times n}$ and $A^\top = A$. Every proof below uses the same single trick:

> compute a scalar of the form $\mathbf{a}^\top A \mathbf{b}$ in **two different ways** — once by letting $A$ act on $\mathbf{b}$, once by moving $A$ onto $\mathbf{a}$ via $A^\top = A$ — and compare.

---

#### Proof 1 — All eigenvalues are real

**Setup.** Let $A\mathbf{v} = \lambda\mathbf{v}$, $\mathbf{v} \neq \mathbf{0}$. We allow $\lambda \in \mathbb{C}$, $\mathbf{v} \in \mathbb{C}^n$ for now.
Let $\bar{\mathbf{v}}$ be the entrywise complex conjugate. Note that

$$\bar{\mathbf{v}}^\top \mathbf{v} = \sum_i |v_i|^2 > 0$$

**Step 1 — let $A$ act on $\mathbf{v}$.**

$$\bar{\mathbf{v}}^\top A \mathbf{v} = \bar{\mathbf{v}}^\top (\lambda \mathbf{v}) = \lambda \,\bar{\mathbf{v}}^\top \mathbf{v}$$

**Step 2 — move $A$ onto $\bar{\mathbf{v}}$.**
$A$ is real, so conjugating $A\mathbf{v} = \lambda\mathbf{v}$ gives $A\bar{\mathbf{v}} = \bar\lambda\bar{\mathbf{v}}$. Then

$$\bar{\mathbf{v}}^\top A \mathbf{v} = (A^\top \bar{\mathbf{v}})^\top \mathbf{v} = (A \bar{\mathbf{v}})^\top \mathbf{v} = (\bar\lambda \bar{\mathbf{v}})^\top \mathbf{v} = \bar\lambda \,\bar{\mathbf{v}}^\top \mathbf{v}$$

**Step 3 — compare.**

$$\lambda \,\bar{\mathbf{v}}^\top \mathbf{v} = \bar\lambda \,\bar{\mathbf{v}}^\top \mathbf{v} \quad\Longrightarrow\quad (\lambda - \bar\lambda)\,\underbrace{\bar{\mathbf{v}}^\top \mathbf{v}}_{>\,0} = 0 \quad\Longrightarrow\quad \lambda = \bar\lambda$$

**Conclusion.** $\lambda \in \mathbb{R}$. Since $A - \lambda I$ is now a *real* singular matrix, its null space contains a real vector, so the eigenvector can be taken real as well. $\blacksquare$

---

#### Proof 2 — Eigenvectors for distinct eigenvalues are orthogonal

**Setup.** Let $A\mathbf{v}_1 = \lambda_1 \mathbf{v}_1$ and $A\mathbf{v}_2 = \lambda_2 \mathbf{v}_2$ with $\lambda_1 \neq \lambda_2$ (both real by Proof 1).

**Step 1 — let $A$ act on $\mathbf{v}_2$.**

$$\mathbf{v}_1^\top A \mathbf{v}_2 = \mathbf{v}_1^\top (\lambda_2 \mathbf{v}_2) = \lambda_2 \,\mathbf{v}_1^\top \mathbf{v}_2$$

**Step 2 — move $A$ onto $\mathbf{v}_1$.**

$$\mathbf{v}_1^\top A \mathbf{v}_2 = \mathbf{v}_1^\top A^\top \mathbf{v}_2 = (A \mathbf{v}_1)^\top \mathbf{v}_2 = (\lambda_1 \mathbf{v}_1)^\top \mathbf{v}_2 = \lambda_1 \,\mathbf{v}_1^\top \mathbf{v}_2$$

**Step 3 — compare.**

$$(\lambda_1 - \lambda_2)\,\mathbf{v}_1^\top \mathbf{v}_2 = 0, \qquad \lambda_1 \neq \lambda_2 \quad\Longrightarrow\quad \mathbf{v}_1^\top \mathbf{v}_2 = 0$$

**Conclusion.** $\mathbf{v}_1 \perp \mathbf{v}_2$. $\blacksquare$

> **What about repeated eigenvalues?** Two eigenvectors sharing the same $\lambda$ need not be orthogonal (any vector in the eigenspace is an eigenvector). But the eigenspace is a subspace, so Gram–Schmidt (§9.3) gives it an orthonormal basis. Proof 3 shows that these bases fill out all of $\mathbb{R}^n$.

---

#### Proof 3 — Always diagonalizable: $A = V\Lambda V^\top$ with $V$ orthogonal (Spectral Theorem)

**Strategy.** Peel off one eigenvector at a time. Induction on $n$.

**Base case $n = 1$.** $A = [a]$, take $V = [1]$, $\Lambda = [a]$.

**Inductive step.** Assume every symmetric $(n-1) \times (n-1)$ matrix is orthogonally diagonalizable.

**Step 1 — grab one eigenpair.**
The characteristic polynomial has a root $\lambda_1 \in \mathbb{C}$; by Proof 1 it is real, with a real unit eigenvector $\mathbf{q}_1$.

**Step 2 — build an orthogonal matrix around it.**
Extend $\mathbf{q}_1$ to an orthonormal basis of $\mathbb{R}^n$ (Gram–Schmidt) and write

$$Q = \begin{bmatrix} \mathbf{q}_1 & Q_2 \end{bmatrix}, \qquad Q_2 \in \mathbb{R}^{n \times (n-1)}, \qquad Q_2^\top \mathbf{q}_1 = \mathbf{0}, \qquad Q^\top Q = I$$

**Step 3 — change basis and watch the off-diagonal blocks vanish.**

$$Q^\top A Q = \begin{bmatrix} \mathbf{q}_1^\top A \mathbf{q}_1 & \mathbf{q}_1^\top A Q_2 \\[4pt] Q_2^\top A \mathbf{q}_1 & Q_2^\top A Q_2 \end{bmatrix}$$

- Top-left: $\mathbf{q}_1^\top A \mathbf{q}_1 = \lambda_1 \mathbf{q}_1^\top \mathbf{q}_1 = \lambda_1$
- Bottom-left: $Q_2^\top A \mathbf{q}_1 = \lambda_1 Q_2^\top \mathbf{q}_1 = \mathbf{0}$
- Top-right: $Q^\top A Q$ is symmetric (because $(Q^\top A Q)^\top = Q^\top A^\top Q = Q^\top A Q$), so it is the transpose of the bottom-left block, also $\mathbf{0}$

$$\therefore\quad Q^\top A Q = \begin{bmatrix} \lambda_1 & \mathbf{0}^\top \\ \mathbf{0} & B \end{bmatrix}, \qquad B := Q_2^\top A Q_2 \;\text{ is symmetric, } (n-1) \times (n-1)$$

**Step 4 — apply the induction hypothesis to $B$.**

$$B = \tilde V \tilde\Lambda \tilde V^\top, \qquad \tilde V \text{ orthogonal}, \; \tilde\Lambda \text{ diagonal}$$

**Step 5 — assemble.**

$$V := Q \begin{bmatrix} 1 & \mathbf{0}^\top \\ \mathbf{0} & \tilde V \end{bmatrix}, \qquad \Lambda := \begin{bmatrix} \lambda_1 & \mathbf{0}^\top \\ \mathbf{0} & \tilde\Lambda \end{bmatrix}$$

$V$ is a product of two orthogonal matrices, so $V^\top V = I$, and

$$V^\top A V = \begin{bmatrix} 1 & \mathbf{0}^\top \\ \mathbf{0} & \tilde V^\top \end{bmatrix} \begin{bmatrix} \lambda_1 & \mathbf{0}^\top \\ \mathbf{0} & B \end{bmatrix} \begin{bmatrix} 1 & \mathbf{0}^\top \\ \mathbf{0} & \tilde V \end{bmatrix} = \begin{bmatrix} \lambda_1 & \mathbf{0}^\top \\ \mathbf{0} & \tilde V^\top B \tilde V \end{bmatrix} = \Lambda$$

**Conclusion.** Multiply by $V$ on the left and $V^\top$ on the right: $A = V \Lambda V^\top$.
The columns of $V$ are $n$ orthonormal eigenvectors; the diagonal of $\Lambda$ holds the real eigenvalues. $\blacksquare$

---

#### Proof 4 — Symmetric bilinear form: $\mathbf{x}^\top A \mathbf{y} = \mathbf{y}^\top A \mathbf{x}$

**($\Rightarrow$) $A$ symmetric implies the form is symmetric.**
$\mathbf{x}^\top A \mathbf{y}$ is a $1 \times 1$ matrix, so it equals its own transpose:

$$\mathbf{x}^\top A \mathbf{y} = (\mathbf{x}^\top A \mathbf{y})^\top = \mathbf{y}^\top A^\top \mathbf{x} = \mathbf{y}^\top A \mathbf{x}$$

**($\Leftarrow$) The form symmetric for all $\mathbf{x}, \mathbf{y}$ implies $A$ symmetric.**
Plug in standard basis vectors $\mathbf{x} = \mathbf{e}_i$, $\mathbf{y} = \mathbf{e}_j$:

$$\mathbf{e}_i^\top A \mathbf{e}_j = a_{ij}, \qquad \mathbf{e}_j^\top A \mathbf{e}_i = a_{ji} \quad\Longrightarrow\quad a_{ij} = a_{ji}$$

**Conclusion.** $A = A^\top \iff \mathbf{x}^\top A \mathbf{y} = \mathbf{y}^\top A \mathbf{x}$ for all $\mathbf{x}, \mathbf{y}$. $\blacksquare$

---

**Where symmetric matrices appear in AI:**

| Matrix | Why symmetric |
|--------|--------------|
| Covariance matrix $\Sigma$ | $\text{Cov}(x_i, x_j) = \text{Cov}(x_j, x_i)$ by definition |
| Hessian $\nabla^2 f$ | Mixed partials are equal: $\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$ |
| Kernel matrix $K$ | $K_{ij} = k(x_i, x_j) = k(x_j, x_i)$ by symmetry of the kernel |
| Attention score matrix | $QK^\top$ is symmetric when $Q = K$ (self-attention) |
| Graph Laplacian $L$ | Encodes undirected graph structure |



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



> **In practice:** never compute $A^{-1}$ explicitly. Use `np.linalg.solve(A, b)` — it is faster and numerically stable.

### 4.3 Gaussian Elimination & Row Reduction

The systematic way to solve $A\mathbf{x} = \mathbf{b}$ by hand and the foundation of most direct solvers.
It uses three **elementary row operations**, none of which change the solution set:

| Operation | Notation |
|-----------|----------|
| Swap two rows | $R_i \leftrightarrow R_j$ |
| Scale a row by a nonzero constant | $R_i \leftarrow cR_i$ |
| Add a multiple of one row to another | $R_i \leftarrow R_i + cR_j$ |

**Forward elimination → Row Echelon Form (REF).** Working left to right, use the pivot in each row to
zero out every entry below it. The result is "staircase" (upper triangular for a square full-rank system):

$$\left[\begin{array}{ccc|c} 2 & 1 & -1 & 8 \\ -3 & -1 & 2 & -11 \\ -2 & 1 & 2 & -3 \end{array}\right]
\longrightarrow
\left[\begin{array}{ccc|c} 2 & 1 & -1 & 8 \\ 0 & \tfrac12 & \tfrac12 & 1 \\ 0 & 0 & -1 & 1 \end{array}\right]$$

**Back substitution.** Solve the triangular system bottom-up: $x_3 = -1$, then
$\tfrac12 x_2 + \tfrac12(-1) = 1 \Rightarrow x_2 = 3$, then $2x_1 + 3 - (-1) = 8 \Rightarrow x_1 = 2$.

**Gauss–Jordan → Reduced Row Echelon Form (RREF).** Keep going: make every pivot $=1$ and clear entries
**above** each pivot too. RREF is **unique** for a given matrix.

$$R = \operatorname{rref}(A): \quad \text{each pivot is 1 and is the only nonzero entry in its column}$$

| Term | Meaning |
|------|---------|
| **Pivot column** | A column containing a leading 1 in the RREF |
| **Free column** | A non-pivot column — its variable is a free parameter |
| $\operatorname{rank}(A)$ | Number of pivots (§8.3) |
| Consistent system | RREF has no row $[\,0\ \cdots\ 0 \mid c\,]$ with $c \neq 0$ |

**Partial pivoting.** Before eliminating with column $k$, swap in the row whose entry in column $k$ has the
**largest absolute value**. Required when the natural pivot is $0$ (otherwise you divide by zero), and it
keeps the multipliers $\leq 1$ so rounding errors do not blow up:

$$\begin{bmatrix} 0 & 1 \\ 1 & 1 \end{bmatrix} \xrightarrow{R_1 \leftrightarrow R_2} \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} \quad \text{— without the swap the first pivot is 0}$$

A pivot column that comes up all-zero (below the current row) with a nonzero right-hand side signals a
**singular** matrix — no unique solution.

**Cost:** $\approx \tfrac{2}{3}n^3$ floating-point operations for forward elimination, then $n^2$ for back substitution.

### 4.4 LU Decomposition

Gaussian elimination, recorded as a factorization. Every step "add $-\ell_{ik}$ times row $k$ to row $i$"
is left-multiplication by a lower-triangular elementary matrix (§4.5). Collecting them:

$$A = LU$$

| Factor | Structure | Contents |
|--------|-----------|----------|
| $L$ | Unit lower triangular ($1$s on the diagonal) | The elimination multipliers $\ell_{ik} = a_{ik}^{(k)}/a_{kk}^{(k)}$ |
| $U$ | Upper triangular | The row-echelon form produced by forward elimination |

With partial pivoting the row swaps are gathered into a permutation matrix $P$:

$$PA = LU$$

**Why factor instead of just solving?** Once you have $L$ and $U$, each new right-hand side costs only
$O(n^2)$ instead of $O(n^3)$:

$$A\mathbf{x} = \mathbf{b} \;\Longleftrightarrow\; L\mathbf{y} = P\mathbf{b}\ \text{(forward sub)},\quad U\mathbf{x} = \mathbf{y}\ \text{(back sub)}$$

This is exactly how `scipy.linalg.lu_factor` / `lu_solve` and `np.linalg.solve` work internally, and how you
build $A^{-1}$ column by column by solving $A\mathbf{x}_k = \mathbf{e}_k$ for each standard basis vector.

**Worked example — factor, then solve.**

$$A = \begin{bmatrix} 2 & 1 & 1 \\ 4 & 3 & 3 \\ 8 & 7 & 9 \end{bmatrix}$$

*Eliminate column 1* ($\ell_{21} = 4/2 = 2$, $\ell_{31} = 8/2 = 4$):

$$R_2 \leftarrow R_2 - 2R_1,\quad R_3 \leftarrow R_3 - 4R_1 \quad\Rightarrow\quad
\begin{bmatrix} 2 & 1 & 1 \\ 0 & 1 & 1 \\ 0 & 3 & 5 \end{bmatrix}$$

*Eliminate column 2* ($\ell_{32} = 3/1 = 3$): $R_3 \leftarrow R_3 - 3R_2 \Rightarrow$ last row becomes $[0, 0, 2]$.

$$L = \begin{bmatrix} 1 & 0 & 0 \\ 2 & 1 & 0 \\ 4 & 3 & 1 \end{bmatrix}, \qquad
U = \begin{bmatrix} 2 & 1 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 2 \end{bmatrix}$$

The multipliers drop straight into $L$ below the diagonal. Check: row 3 of $LU$ is
$4[2,1,1] + 3[0,1,1] + [0,0,2] = [8,7,9]$ ✓.

*Solve $A\mathbf{x} = \mathbf{b}$ with $\mathbf{b} = [1, 1, 3]^\top$.* Forward-substitute $L\mathbf{y} = \mathbf{b}$:

$$y_1 = 1,\quad y_2 = 1 - 2(1) = -1,\quad y_3 = 3 - 4(1) - 3(-1) = 2$$

Back-substitute $U\mathbf{x} = \mathbf{y}$:

$$x_3 = \tfrac{2}{2} = 1,\quad x_2 = -1 - 1 = -2,\quad x_1 = \tfrac{1 - (-2) - 1}{2} = 1 \;\Rightarrow\; \mathbf{x} = \begin{bmatrix} 1 \\ -2 \\ 1 \end{bmatrix}$$

And $\det(A) = \prod_i U_{ii} = 2 \cdot 1 \cdot 2 = 4$ (no row swaps → $+$ sign).

**With pivoting — $PA = LU$.**

$$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}: \quad |3| > |1| \text{ in column 1, so swap rows.}\quad
P = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$$

$$PA = \begin{bmatrix} 3 & 4 \\ 1 & 2 \end{bmatrix}, \quad \ell_{21} = \tfrac{1}{3} \;\Rightarrow\;
L = \begin{bmatrix} 1 & 0 \\ \tfrac13 & 1 \end{bmatrix},\quad U = \begin{bmatrix} 3 & 4 \\ 0 & \tfrac23 \end{bmatrix}$$

**Cholesky — $A = LL^\top$ for symmetric positive definite $A$.**

$$A = \begin{bmatrix} 4 & 2 \\ 2 & 3 \end{bmatrix} \quad (\text{leading minors } 4 > 0,\ \det = 8 > 0 \Rightarrow \text{SPD})$$

$$\ell_{11} = \sqrt{4} = 2,\quad \ell_{21} = \tfrac{2}{\ell_{11}} = 1,\quad \ell_{22} = \sqrt{3 - \ell_{21}^2} = \sqrt{2}$$

$$L = \begin{bmatrix} 2 & 0 \\ 1 & \sqrt{2} \end{bmatrix}, \qquad
LL^\top = \begin{bmatrix} 4 & 2 \\ 2 & 1 + 2 \end{bmatrix} = A \ \checkmark$$

**Related factorizations:**

| Matrix type | Factorization | Note |
|-------------|---------------|------|
| General square | $PA = LU$ | Partial pivoting for stability |
| Symmetric positive definite | $A = LL^\top$ (**Cholesky**) | Half the cost, no pivoting needed (§10) |
| Symmetric indefinite | $A = LDL^\top$ | $D$ block-diagonal |
| $\det(A)$ | $\pm\prod_i U_{ii}$ | Sign from the number of row swaps |

### 4.5 Elementary Matrices & Row Equivalence

Each elementary row operation **is** left-multiplication by an **elementary matrix** $E$ — the identity with
one small modification:

| Operation | Elementary matrix $E$ | $E^{-1}$ |
|-----------|----------------------|----------|
| Swap rows $i, j$ | $I$ with rows $i, j$ swapped (a permutation) | itself |
| Scale row $i$ by $c \neq 0$ | $I$ with $E_{ii} = c$ | scale by $1/c$ |
| Add $c\cdot$(row $j$) to row $i$ | $I$ with $E_{ij} = c$ | same with $-c$ |

Every elementary matrix is invertible, and its inverse is elementary of the same type. Running row reduction
is therefore

$$E_k E_{k-1} \cdots E_1 A = R, \qquad \text{so } EA = R \text{ with } E = E_k \cdots E_1 \text{ invertible.}$$

**Row equivalence.** $A$ and $B$ are **row equivalent** when any of these equivalent conditions holds:

- $B = EA$ for some invertible $E$;
- $A$ and $B$ have the **same RREF**;
- $A$ and $B$ have the **same row space** (§8.4).

> Equal rank is **necessary but not sufficient**: $\begin{bmatrix}1&0&1\\0&1&1\end{bmatrix}$ and
> $\begin{bmatrix}1&0&0\\0&1&1\end{bmatrix}$ both have rank 2 but different RREFs, so they are **not** row equivalent.

The same idea on columns gives $AF = C$ with $F$ invertible (column operations); combining both,
$EAF = \begin{bmatrix} I_r & 0 \\ 0 & 0 \end{bmatrix}$ — the **rank normal form** of any matrix.

**Example.** $A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$ has rank 1. Row-reduce with
$E = \begin{bmatrix} 1 & 0 \\ -2 & 1 \end{bmatrix}$ ($R_2 \leftarrow R_2 - 2R_1$), then clear the second column with
$F = \begin{bmatrix} 1 & -2 \\ 0 & 1 \end{bmatrix}$ ($C_2 \leftarrow C_2 - 2C_1$):

$$EA = \begin{bmatrix} 1 & 2 \\ 0 & 0 \end{bmatrix}, \qquad
EAF = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} I_1 & 0 \\ 0 & 0 \end{bmatrix}$$

---

## 5. Determinant

### 5.1 Definition & Geometric Meaning

The **determinant** of a square matrix $A \in \mathbb{R}^{n \times n}$ is a single scalar that captures how the linear transformation $A$ scales volume.

$$\det(A) \in \mathbb{R}$$

**2×2 case:**

$$A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}, \qquad \det(A) = ad - bc$$

**Geometric meaning:** $|\det(A)|$ is the factor by which $A$ scales areas (2D) or volumes (3D).



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

The determinant is the **product of all eigenvalues**. If any eigenvalue is zero, the determinant is zero — meaning the matrix crushes at least one direction flat. (See §6.1 for how eigenvalues are found via $\det(A - \lambda I) = 0$.)



> **One-line summary:** the determinant measures how much a matrix scales volume. Zero determinant = the matrix destroys information by collapsing space.

### 5.3 Cofactor (Laplace) Expansion

How to compute $\det(A)$ for a general $n \times n$ matrix — by recursion on size.

| Term | Definition |
|------|------------|
| **Minor** $M_{ij}$ | $\det$ of the $(n{-}1)\times(n{-}1)$ matrix left after deleting row $i$ and column $j$ |
| **Cofactor** $C_{ij}$ | $(-1)^{i+j} M_{ij}$ — the minor with a checkerboard sign |

**Laplace expansion** along *any* row $i$ (or any column $j$):

$$\det(A) = \sum_{j=1}^{n} a_{ij} C_{ij} = \sum_{j=1}^{n} (-1)^{i+j}\, a_{ij}\, M_{ij}$$

The sign pattern:

$$\begin{bmatrix} + & - & + & \cdots \\ - & + & - & \\ + & - & + & \\ \vdots & & & \ddots \end{bmatrix}$$

**3×3 example — expand along the first row:**

$$\det\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 10 \end{bmatrix}
= 1\det\begin{bmatrix}5&6\\8&10\end{bmatrix} - 2\det\begin{bmatrix}4&6\\7&10\end{bmatrix} + 3\det\begin{bmatrix}4&5\\7&8\end{bmatrix}$$
$$= 1(50-48) - 2(40-42) + 3(32-35) = 2 + 4 - 9 = -3$$

**Base case:** $\det[a] = a$ for a $1\times1$ matrix. Choosing a row or column with many zeros minimizes the work.

**Cost & practice:** naive cofactor expansion is $O(n!)$ — only for tiny or symbolic matrices. For numbers,
reduce to triangular form by elimination (§4.3) and multiply the pivots: $\det(A) = (-1)^{\#\text{swaps}}\prod_i U_{ii}$, which is $O(n^3)$.

**Two consequences of the cofactor formula:**

$$\text{adjugate: } \operatorname{adj}(A)_{ij} = C_{ji}, \qquad A^{-1} = \frac{1}{\det(A)}\operatorname{adj}(A)$$

$$\text{Cramer's rule: } x_i = \frac{\det(A_i)}{\det(A)}, \quad A_i = A \text{ with column } i \text{ replaced by } \mathbf{b}$$

Both are theoretically clean but computationally far slower than elimination — use them for proofs and $2\times2$ / $3\times3$ hand work, not code.

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

### 6.3 Eigendecomposition

For a diagonalizable matrix $A$ with $n$ independent eigenvectors:

$$A = V \Lambda V^{-1}$$

| Term | Shape | Meaning |
|------|-------|---------|
| $V$ | $n \times n$ | Columns are eigenvectors — the natural axes of $A$ |
| $\Lambda$ | $n \times n$ diagonal | Diagonal entries are eigenvalues $\lambda_1, \ldots, \lambda_n$ |
| $V^{-1}$ | $n \times n$ | Change of basis back to original coordinates |

**Geometric reading:** $V^{-1}$ rotates the world so the eigenvectors become the coordinate axes, $\Lambda$
stretches each axis by its eigenvalue, and $V$ rotates back. In the eigenbasis, $A$ is just a diagonal
(axis-wise) scaling.

**Worked example.**

$$A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$

*Eigenvalues* — solve $\det(A - \lambda I) = (2-\lambda)^2 - 1 = (\lambda - 1)(\lambda - 3) = 0$:
$\lambda_1 = 3,\ \lambda_2 = 1$.

*Eigenvectors* — solve $(A - \lambda I)\mathbf{v} = \mathbf{0}$:

$$\lambda_1 = 3:\ \begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix}\mathbf{v} = \mathbf{0} \Rightarrow \mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \qquad
\lambda_2 = 1:\ \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}\mathbf{v} = \mathbf{0} \Rightarrow \mathbf{v}_2 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$$

$$V = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix},\quad
\Lambda = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix},\quad
V^{-1} = \begin{bmatrix} \tfrac12 & \tfrac12 \\ \tfrac12 & -\tfrac12 \end{bmatrix}$$

Check: $V\Lambda V^{-1} = \begin{bmatrix} 3 & 1 \\ 3 & -1 \end{bmatrix}\begin{bmatrix} \tfrac12 & \tfrac12 \\ \tfrac12 & -\tfrac12 \end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ ✓.
Note $\det A = 2 = 3 \cdot 1 = \lambda_1 \lambda_2$ and $\operatorname{tr} A = 4 = 3 + 1$.

For **symmetric** $A$ (as here): eigenvectors are always orthogonal, so normalizing them gives an orthogonal
$Q$ with $V^{-1} = Q^\top$:

$$A = Q \Lambda Q^\top \qquad \text{(spectral decomposition)}, \qquad
Q = \tfrac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$$

Equivalently, $A$ is a weighted sum of orthogonal rank-1 projectors:

$$A = \sum_i \lambda_i\, \mathbf{q}_i \mathbf{q}_i^\top
= 3 \cdot \tfrac12\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} + 1 \cdot \tfrac12\begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$$

For a **non-symmetric** matrix such as $\begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix}$, the eigenvalues are still
read off (here $2, 3$) but $V = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ is not orthogonal, so
$V^{-1} = \begin{bmatrix} 1 & -1 \\ 0 & 1 \end{bmatrix} \neq V^\top$.


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

**How to compute it:** the $\sigma_i^2$ are the eigenvalues of $A^\top A$ (or $AA^\top$), the columns of $V$
are the eigenvectors of $A^\top A$, and then $\mathbf{u}_i = A\mathbf{v}_i / \sigma_i$.

**Worked example** ($2 \times 3$, so a genuinely rectangular map):

$$A = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$

$$AA^\top = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix} \;\Rightarrow\; \text{eigenvalues } 2, 1
\;\Rightarrow\; \sigma_1 = \sqrt{2},\ \sigma_2 = 1, \quad U = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$

$$\mathbf{v}_1 = \frac{A^\top \mathbf{u}_1}{\sigma_1} = \tfrac{1}{\sqrt2}\begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix},\quad
\mathbf{v}_2 = \frac{A^\top \mathbf{u}_2}{\sigma_2} = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix},\quad
\mathbf{v}_3 = \tfrac{1}{\sqrt2}\begin{bmatrix} 1 \\ 0 \\ -1 \end{bmatrix} \ (\text{spans } N(A))$$

$$\Sigma = \begin{bmatrix} \sqrt2 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}, \qquad
V = \begin{bmatrix} 1/\sqrt2 & 0 & 1/\sqrt2 \\ 0 & 1 & 0 \\ 1/\sqrt2 & 0 & -1/\sqrt2 \end{bmatrix}$$

Check: $\Sigma V^\top = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \end{bmatrix} = A$ (since $U = I$) ✓. The third
right singular vector has $\sigma_3 = 0$ — it is exactly the null-space direction $A$ collapses.

### 7.2 Geometric Meaning

Every linear transformation $A$ can be decomposed into three steps — **rotate, scale, rotate**:

$$A\mathbf{x} = \underbrace{U}_{\text{rotate}}\ \underbrace{\Sigma}_{\text{scale axes}}\ \underbrace{V^\top}_{\text{rotate}}\ \mathbf{x}$$

Geometrically: $A$ maps the unit sphere to an ellipsoid. $V^\top$ picks the input axes that map to the
ellipsoid's principal axes, $\Sigma$ stretches them to lengths $\sigma_1, \sigma_2, \ldots$, and $U$ orients
the ellipsoid in the output space.

The singular values $\sigma_i$ tell you how much $A$ stretches space along each direction. Large $\sigma_i$ = important direction; small $\sigma_i \approx 0$ = negligible direction.


### 7.3 Low-Rank Approximation

Keep only the top $k$ singular values — best possible rank-$k$ approximation of $A$:

$$A \approx A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top = U_k \Sigma_k V_k^\top$$

| Term | Meaning |
|------|---------|
| $\sigma_i$ | How important the $i$-th component is |
| $\mathbf{u}_i \mathbf{v}_i^\top$ | Rank-1 matrix — outer product of two vectors |
| $k \ll \min(m,n)$ | Keep only the most important components |

**Worked example — a rank-1 matrix has exactly one SVD layer:**

$$A = \begin{bmatrix} 2 & 2 \\ 1 & 1 \end{bmatrix}, \quad
A^\top A = \begin{bmatrix} 5 & 5 \\ 5 & 5 \end{bmatrix} \;\Rightarrow\; \text{eigenvalues } 10, 0
\;\Rightarrow\; \sigma_1 = \sqrt{10},\ \sigma_2 = 0$$

$$\mathbf{v}_1 = \tfrac{1}{\sqrt2}\begin{bmatrix} 1 \\ 1 \end{bmatrix},\quad
\mathbf{u}_1 = \frac{A\mathbf{v}_1}{\sigma_1} = \tfrac{1}{\sqrt5}\begin{bmatrix} 2 \\ 1 \end{bmatrix}$$

$$A = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^\top
= \sqrt{10}\cdot\tfrac{1}{\sqrt5}\begin{bmatrix} 2 \\ 1 \end{bmatrix}\cdot\tfrac{1}{\sqrt2}\begin{bmatrix} 1 & 1 \end{bmatrix}
= \begin{bmatrix} 2 & 2 \\ 1 & 1 \end{bmatrix} \ \checkmark$$

The single nonzero singular value carries the whole matrix; $\sigma_2 = 0$ means dropping to rank 1 loses nothing.
For a full-rank matrix, truncating to rank $k$ discards $\sum_{i>k}\sigma_i^2$ worth of squared "energy" — the
Eckart–Young error $\|A - A_k\|_F^2$.

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

| rank(A) | $\dim(\text{null}(A))$ | Meaning |
|-----------|------------------------|---------|
| $n$ (full) | $0$ | Trivial null space — only $\mathbf{x}=\mathbf{0}$ maps to $\mathbf{0}$ — $A$ is invertible |
| $n-1$ | $1$ | One direction collapses — $A$ is singular |
| $0$ | $n$ | Everything maps to $\mathbf{0}$ — $A$ is the zero matrix |

**Concrete example:**

$$A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$$

Column 2 = $2 \times$ Column 1 → rank = 1. The null space is all $\mathbf{x}$ satisfying $x_1 + 2x_2 = 0$:

$$\text{null}(A) = \left\{ t \begin{bmatrix} -2 \\ 1 \end{bmatrix} \mid t \in \mathbb{R} \right\} \quad \text{(a line through the origin)}$$



**In AI:** if a weight matrix $W$ has a large null space, many input directions are completely ignored — the layer has low effective rank. LoRA exploits this: large pretrained weight matrices tend to be approximately low-rank, so you only need to learn a small rank update $\Delta W = AB$.

---

### 8.4 The Four Fundamental Subspaces

Every matrix $A \in \mathbb{R}^{m \times n}$ of rank $r$ has four associated subspaces. Together they describe
everything the linear map $\mathbf{x} \mapsto A\mathbf{x}$ does.

| Subspace | Symbol | Lives in | Dimension | Basis from row reduction |
|----------|--------|----------|-----------|--------------------------|
| **Column space** (range/image) | $C(A)$ | $\mathbb{R}^m$ | $r$ | The **original** columns of $A$ at pivot positions |
| **Null space** (kernel) | $N(A)$ | $\mathbb{R}^n$ | $n - r$ | One vector per free column of $\operatorname{rref}(A)$ |
| **Row space** | $C(A^\top)$ | $\mathbb{R}^n$ | $r$ | The nonzero rows of $\operatorname{rref}(A)$ |
| **Left null space** | $N(A^\top)$ | $\mathbb{R}^m$ | $m - r$ | $N(A^\top)$, or the rows of $E$ (from $EA=R$) that zero out |

**Fundamental Theorem of Linear Algebra** — within each ambient space the two subspaces are **orthogonal
complements**:

$$C(A^\top) \perp N(A) \quad \text{and} \quad C(A^\top) \oplus N(A) = \mathbb{R}^n$$
$$C(A) \perp N(A^\top) \quad \text{and} \quad C(A) \oplus N(A^\top) = \mathbb{R}^m$$

$C(A^\top) \perp N(A)$ is immediate: if $A\mathbf{x} = \mathbf{0}$ then every row of $A$ dotted with $\mathbf{x}$ is $0$.

**Reading a null-space basis off the RREF.** For each free column, set that free variable to $1$, the other
free variables to $0$, and solve the pivot rows:

$$\operatorname{rref}(A) = \begin{bmatrix} 1 & 2 & 0 & 2 \\ 0 & 0 & 1 & 1 \\ 0 & 0 & 0 & 0 \end{bmatrix}
\quad\Rightarrow\quad
N(A) = \operatorname{span}\!\left\{ \begin{bmatrix}-2\\1\\0\\0\end{bmatrix},\ \begin{bmatrix}-2\\0\\-1\\1\end{bmatrix} \right\}$$

(columns 2 and 4 are free; $r = 2$, $n - r = 2$). The pivot columns $1, 3$ of the **original** $A$ form a basis
for $C(A)$; the two nonzero RREF rows form a basis for $C(A^\top)$.

**Existence and uniqueness of solutions to $A\mathbf{x} = \mathbf{b}$:**

| Condition | Consequence |
|-----------|-------------|
| $\mathbf{b} \in C(A)$ (equivalently $\mathbf{b} \perp N(A^\top)$) | At least one solution exists |
| $N(A) = \{\mathbf{0}\}$ (full column rank) | Solution is unique if it exists |
| $\dim N(A) > 0$ | Solutions form the coset $\mathbf{x}_p + N(A)$ (particular + homogeneous) |

**Example — incidence matrix of a graph.** For a connected graph with $n$ nodes and $m$ edges, let each row
of $A$ be an edge $i \to j$ with $-1$ in column $i$, $+1$ in column $j$:

- $N(A) = \operatorname{span}\{[1,1,\ldots,1]^\top\}$, dimension $1$ — adding a constant to every node potential changes no edge difference.
- $\operatorname{rank}(A) = n - 1$; the row space is the zero-sum node vectors.
- $N(A^\top)$ has dimension $m - n + 1$ — the number of **independent cycles** (Euler's formula). Its vectors are the loop currents that satisfy Kirchhoff's voltage law.

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

**In AI:** attention computes projections; least squares regression minimizes $\|A\mathbf{x} - \mathbf{b}\|_2^2$ via projection (full treatment in §13).

### 9.3 Gram–Schmidt Orthogonalization

Turns a linearly independent list $\mathbf{a}_1, \ldots, \mathbf{a}_n$ into an **orthonormal** list
$\mathbf{q}_1, \ldots, \mathbf{q}_n$ that spans the same subspace — and, step by step,
$\operatorname{span}(\mathbf{q}_1, \ldots, \mathbf{q}_j) = \operatorname{span}(\mathbf{a}_1, \ldots, \mathbf{a}_j)$ for every $j$.

$$\tilde{\mathbf{q}}_j = \mathbf{a}_j - \sum_{k<j} (\mathbf{q}_k^\top \mathbf{a}_j)\,\mathbf{q}_k, \qquad \mathbf{q}_j = \frac{\tilde{\mathbf{q}}_j}{\|\tilde{\mathbf{q}}_j\|}$$

Each sum term is the projection of $\mathbf{a}_j$ onto an already-built axis (§9.2); subtracting them all
leaves the part of $\mathbf{a}_j$ orthogonal to everything before it. If $\|\tilde{\mathbf{q}}_j\| = 0$, then
$\mathbf{a}_j$ was in the span of the earlier vectors — the inputs were **not** independent.

**Concrete example:**

$$\mathbf{a}_1 = \begin{bmatrix}1\\1\\0\end{bmatrix},\ \mathbf{a}_2 = \begin{bmatrix}1\\0\\1\end{bmatrix}
\;\Rightarrow\;
\mathbf{q}_1 = \tfrac{1}{\sqrt2}\begin{bmatrix}1\\1\\0\end{bmatrix},\quad
\tilde{\mathbf{q}}_2 = \mathbf{a}_2 - \tfrac12\begin{bmatrix}1\\1\\0\end{bmatrix} = \begin{bmatrix}\tfrac12\\-\tfrac12\\1\end{bmatrix},\quad
\mathbf{q}_2 = \tfrac{1}{\sqrt6}\begin{bmatrix}1\\-1\\2\end{bmatrix}$$

**Classical vs modified.** Mathematically identical, numerically different:

| Variant | Update rule | Stability |
|---------|-------------|-----------|
| Classical (CGS) | Subtract all projections onto the **original** $\mathbf{a}_j$ at once | Loses orthogonality under rounding |
| Modified (MGS) | Subtract each projection from the **running** vector, one $\mathbf{q}_k$ at a time | Much better — preferred in code |

### 9.4 QR Factorization

Gram–Schmidt on the columns of $A$, written as a matrix product:

$$A = QR, \qquad A \in \mathbb{R}^{m \times n}\ (m \geq n,\ \text{full column rank})$$

| Factor | Shape | Meaning |
|--------|-------|---------|
| $Q$ | $m \times n$ | Orthonormal columns ($Q^\top Q = I_n$) — the $\mathbf{q}_j$ from Gram–Schmidt |
| $R$ | $n \times n$ | Upper triangular, $R_{kj} = \mathbf{q}_k^\top \mathbf{a}_j$, $R_{jj} = \|\tilde{\mathbf{q}}_j\| > 0$ |

$R$ is upper triangular precisely *because* $\mathbf{q}_k$ (built from $\mathbf{a}_1, \ldots, \mathbf{a}_k$) is
orthogonal to $\mathbf{a}_j$ for $j < k$. Since $Q^\top Q = I$, we get $R = Q^\top A$.

**Worked example** — the same columns as the Gram–Schmidt example above, now assembled into $Q$ and $R$:

$$A = \begin{bmatrix} 1 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}
\quad(\mathbf{a}_1 = [1,1,0]^\top,\ \mathbf{a}_2 = [1,0,1]^\top)$$

$$R_{11} = \|\mathbf{a}_1\| = \sqrt2, \quad
R_{12} = \mathbf{q}_1^\top \mathbf{a}_2 = \tfrac{1}{\sqrt2}, \quad
R_{22} = \|\tilde{\mathbf{q}}_2\| = \sqrt{\tfrac32} = \tfrac{\sqrt6}{2}$$

$$Q = \begin{bmatrix} 1/\sqrt2 & 1/\sqrt6 \\ 1/\sqrt2 & -1/\sqrt6 \\ 0 & 2/\sqrt6 \end{bmatrix}, \qquad
R = \begin{bmatrix} \sqrt2 & 1/\sqrt2 \\ 0 & \sqrt6/2 \end{bmatrix}$$

Check column 2 of $QR$: $\tfrac{1}{\sqrt2}\mathbf{q}_1 + \tfrac{\sqrt6}{2}\mathbf{q}_2
= [\tfrac12, \tfrac12, 0]^\top + [\tfrac12, -\tfrac12, 1]^\top = [1, 0, 1]^\top = \mathbf{a}_2$ ✓.

- **Reduced** QR: $Q$ is $m \times n$, $R$ is $n \times n$ (above).
- **Full** QR: $Q$ is $m \times m$ orthogonal, $R$ is $m \times n$ with a zero block below row $n$.

| Use | How |
|-----|-----|
| Solve square $A\mathbf{x} = \mathbf{b}$ | $R\mathbf{x} = Q^\top \mathbf{b}$, then back substitution (§4.3) |
| Least squares, tall $A$ | Same formula — see §13.3 |
| $\det(A)$ (square) | $\pm\prod_i R_{ii}$ |
| Eigenvalues | The **QR algorithm**: repeatedly $A \leftarrow RQ$ from $A = QR$ converges to triangular |

**Numerical note.** In practice QR is built with **Householder reflections** or **Givens rotations** rather
than Gram–Schmidt — they are backward stable. QR-based least squares avoids squaring the condition number,
which $A^\top A$ in the normal equations does ($\kappa(A^\top A) = \kappa(A)^2$).

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

**Proof — PD $\iff$ all eigenvalues are positive**

Throughout, $A$ is symmetric, so by §2.5 its eigenvalues are real and $A = V\Lambda V^\top$ with $V$ orthogonal.

---

#### ($\Rightarrow$) PD implies every eigenvalue is positive

**Setup.** Let $\lambda$ be any eigenvalue with eigenvector $\mathbf{v} \neq \mathbf{0}$, so $A\mathbf{v} = \lambda\mathbf{v}$.

**Step 1 — plug the eigenvector into the quadratic form.**

$$\mathbf{v}^\top A \mathbf{v} = \mathbf{v}^\top (\lambda \mathbf{v}) = \lambda\, \mathbf{v}^\top \mathbf{v} = \lambda\, \|\mathbf{v}\|^2$$

**Step 2 — use the PD hypothesis.**
$\mathbf{v} \neq \mathbf{0}$, so PD gives $\mathbf{v}^\top A \mathbf{v} > 0$. Hence

$$\lambda\, \underbrace{\|\mathbf{v}\|^2}_{>\,0} > 0 \quad\Longrightarrow\quad \lambda > 0$$

**Conclusion.** Every eigenvalue is strictly positive. $\blacksquare$

> **Intuition:** the quadratic form $\mathbf{x}^\top A \mathbf{x}$ measures how much $A$ "stretches" $\mathbf{x}$ *along itself*. On an eigenvector, $A$ acts as pure scaling by $\lambda$, so the stretch is exactly $\lambda \|\mathbf{v}\|^2$. If that must be positive, $\lambda$ must be positive.

---

#### ($\Leftarrow$) All eigenvalues positive implies PD

**Setup.** Assume $\lambda_1, \ldots, \lambda_n > 0$ and $A = V\Lambda V^\top$. Take any $\mathbf{x} \neq \mathbf{0}$.

**Step 1 — change to the eigenvector basis.**
Let $\mathbf{y} = V^\top \mathbf{x}$. Since $V$ is orthogonal, $\mathbf{y} \neq \mathbf{0}$ (because $\mathbf{x} = V\mathbf{y}$ and $V\mathbf{0} = \mathbf{0}$).

**Step 2 — the quadratic form becomes a weighted sum of squares.**

$$\mathbf{x}^\top A \mathbf{x} = \mathbf{x}^\top V \Lambda V^\top \mathbf{x} = (V^\top \mathbf{x})^\top \Lambda (V^\top \mathbf{x}) = \mathbf{y}^\top \Lambda \mathbf{y} = \sum_{i=1}^n \lambda_i\, y_i^2$$

**Step 3 — every term is non-negative and at least one is positive.**
Each $\lambda_i > 0$ and $y_i^2 \geq 0$, so every term is $\geq 0$. Since $\mathbf{y} \neq \mathbf{0}$, some $y_j \neq 0$, and that term $\lambda_j y_j^2 > 0$. Hence

$$\mathbf{x}^\top A \mathbf{x} = \sum_{i=1}^n \lambda_i\, y_i^2 > 0$$

**Conclusion.** $\mathbf{x}^\top A \mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$, i.e. $A$ is PD. $\blacksquare$

> **PSD version:** replace every strict inequality with $\geq$. The same two arguments give $A$ is PSD $\iff$ all $\lambda_i \geq 0$.

---

**Why AI cares:**
- The **Hessian** of the loss being PD at a critical point → strict local minimum; PD everywhere → convex with a unique minimum ([multivariable_calculus.md](multivariable_calculus.md) §7–8)
- **Covariance matrices** are always PSD — eigenvalues represent variance along each direction
- **Kernel matrices** in SVMs must be PSD
- Adam optimizer uses a diagonal approximation of the PD Hessian



---

## 11. Gradients & Jacobians

> **Scope:** this section defines the gradient, Jacobian, and Hessian as linear-algebra *objects* — their shapes and the identities used in §12. The *calculus* — why the gradient is the steepest-ascent direction, directional derivatives, the chain rule as function composition, Taylor expansion, and optimization — lives in [multivariable_calculus.md](multivariable_calculus.md) §2–9.

### 11.1 Gradient as a Vector

For a scalar function $f: \mathbb{R}^n \to \mathbb{R}$, the gradient is the vector of partial derivatives:

$$\nabla_{\mathbf{x}} f = \begin{bmatrix} \partial f / \partial x_1 \\ \partial f / \partial x_2 \\ \vdots \\ \partial f / \partial x_n \end{bmatrix} \in \mathbb{R}^n$$

It points in the direction of **steepest increase** (derived in [multivariable_calculus.md](multivariable_calculus.md) §3.2); gradient descent steps opposite to it:

$$\mathbf{x} \leftarrow \mathbf{x} - \alpha \nabla_\mathbf{x} f(\mathbf{x})$$

**Common gradient identities** (used throughout §12):

| Function | Gradient |
|----------|----------|
| $f(\mathbf{x}) = \mathbf{a}^\top \mathbf{x}$ | $\nabla f = \mathbf{a}$ |
| $f(\mathbf{x}) = \mathbf{x}^\top A \mathbf{x}$ | $\nabla f = (A + A^\top)\mathbf{x} = 2A\mathbf{x}$ (if $A$ symmetric) |
| $f(\mathbf{x}) = \|\mathbf{x}\|_2^2$ | $\nabla f = 2\mathbf{x}$ |
| $f(W) = \mathbf{a}^\top W \mathbf{b}$ | $\nabla_W f = \mathbf{a}\mathbf{b}^\top$ |

### 11.2 Jacobian Matrix

For a vector-valued function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian contains all first partials. **This document uses denominator layout** — shape $n \times m$, with $J_{ij} = \partial f_j / \partial x_i$:

$$J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix} \partial f_1/\partial x_1 & \cdots & \partial f_m/\partial x_1 \\ \vdots & \ddots & \vdots \\ \partial f_1/\partial x_n & \cdots & \partial f_m/\partial x_n \end{bmatrix} \in \mathbb{R}^{n \times m}$$

| Term | Meaning |
|------|---------|
| $J_{ij}$ | How output $j$ changes when input $i$ changes |
| $J \in \mathbb{R}^{n \times m}$ | denominator shape ($\mathbf{x}$) × numerator shape ($\mathbf{f}$) |
| $\det(J)$ | Local volume scaling factor — see [multivariable_calculus.md](multivariable_calculus.md) §6.3 |

> **Convention bridge:** [multivariable_calculus.md](multivariable_calculus.md) §6 uses **numerator layout** ($m \times n$, the math/robotics standard, where row $i$ is $\nabla f_i^\top$ and $J$ reads directly as the local linear map). This ML section uses **denominator layout** ($n \times m$) so gradients match parameter shapes for the update step. The two are transposes; each doc is internally consistent.

**In AI:** backpropagation computes Jacobian-vector products via the chain rule (see §12.4):

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = J \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{f}}$$

For the geometric picture (local linear map, inverse function theorem) and the robot manipulator Jacobian, see [multivariable_calculus.md](multivariable_calculus.md) §6.2–6.4.

### 11.3 Hessian Matrix

For $f: \mathbb{R}^n \to \mathbb{R}$, the Hessian is the matrix of second partials — equivalently the Jacobian of the gradient:

$$H = \nabla^2 f = \begin{bmatrix} \partial^2 f/\partial x_1^2 & \partial^2 f/\partial x_1 \partial x_2 & \cdots \\ \partial^2 f/\partial x_2 \partial x_1 & \partial^2 f/\partial x_2^2 & \cdots \\ \vdots & & \ddots \end{bmatrix} \in \mathbb{R}^{n \times n}$$

$H$ is **symmetric** (equality of mixed partials), so it has real eigenvalues and orthogonal eigenvectors — §2.5, §6.

- Its eigenvalues are the curvatures of $f$; their **signs classify critical points** (PD → min, ND → max, mixed → saddle). Full treatment in [multivariable_calculus.md](multivariable_calculus.md) §7–8.
- The **condition number** $\kappa(H) = \lambda_{\max}/\lambda_{\min}$ (§7.4) governs optimizer behavior: large $\kappa$ → a narrow ravine where plain gradient descent zig-zags, motivating momentum / Adam / preconditioning.

---

## 12. Matrix Derivatives

Matrix derivatives are the engine of backpropagation. Every gradient update in a neural network is a matrix derivative computed through the chain rule. This section is the **ML-facing reference**: denominator layout, shape bookkeeping, and worked numerical examples. For the underlying calculus — partial derivatives, the multivariable chain rule, Taylor expansion, and optimization theory — see [multivariable_calculus.md](multivariable_calculus.md).

### 12.1 Layout Convention

**This document uses denominator layout**, because it makes gradient descent work without any transposing: if $\mathbf{w}$ has shape $(n,)$ then $\partial L/\partial\mathbf{w}$ also has shape $(n,)$, so the update $\mathbf{w} \leftarrow \mathbf{w} - \eta\,\nabla_\mathbf{w} L$ is shape-consistent. PyTorch and NumPy follow this convention.

> This is the **opposite** of the numerator layout used for the Jacobian in [multivariable_calculus.md](multivariable_calculus.md) §6. Denominator layout is the right default for ML (gradients mirror parameters); numerator layout is the right default for kinematics and analysis (the Jacobian is read directly as the local linear map $\mathbf{f}(\mathbf{x}_0) + J\Delta\mathbf{x}$). Same numbers, transposed arrangement.

#### The shape table

| Derivative | Numerator shape | Denominator shape | Result shape | Name |
|------------|----------------|-------------------|--------------|------|
| $\partial y / \partial x$ | scalar | scalar | scalar | ordinary derivative |
| $\partial y / \partial \mathbf{x}$ | scalar | $(n,)$ | $(n,)$ | gradient vector |
| $\partial \mathbf{y} / \partial x$ | $(m,)$ | scalar | $(m,)$ | tangent vector |
| $\partial \mathbf{y} / \partial \mathbf{x}$ | $(m,)$ | $(n,)$ | $(n \times m)$ | Jacobian matrix |
| $\partial y / \partial X$ | scalar | $(m \times n)$ | $(m \times n)$ | matrix gradient |

> **Rule of thumb (denominator layout):** the result always has the **same shape as the denominator** — the thing you differentiate with respect to.

#### Row-by-row intuition with examples

**Row 1 — scalar / scalar:** ordinary calculus.

$$\frac{\partial}{\partial x}(x^2) = 2x \qquad \text{scalar in, scalar out}$$

**Row 2 — scalar / vector:** the most common case in ML — loss $L$ differentiated w.r.t. a weight vector $\mathbf{w} \in \mathbb{R}^n$.

$$\frac{\partial L}{\partial \mathbf{w}} = \begin{bmatrix}\partial L/\partial w_1 \\ \vdots \\ \partial L/\partial w_n\end{bmatrix} \in \mathbb{R}^n$$

One entry per weight, telling you how much $L$ changes if you nudge that weight. Shape matches $\mathbf{w}$, so the gradient descent step $\mathbf{w} \leftarrow \mathbf{w} - \eta\,\nabla L$ is always valid.

**Row 3 — vector / scalar:** a vector-valued quantity differentiated w.r.t. a scalar — e.g., position $\mathbf{p}(t) \in \mathbb{R}^3$ differentiated w.r.t. time $t$.

$$\frac{\partial \mathbf{p}}{\partial t} = \begin{bmatrix}\dot{p}_x \\ \dot{p}_y \\ \dot{p}_z\end{bmatrix} \in \mathbb{R}^3 \qquad \text{(velocity vector)}$$

**Row 4 — vector / vector:** produces the **Jacobian** — a matrix whose $(i,j)$ entry is $\partial y_j / \partial x_i$.

$$\mathbf{y} \in \mathbb{R}^m,\quad \mathbf{x} \in \mathbb{R}^n \quad\Rightarrow\quad J = \frac{\partial \mathbf{y}}{\partial \mathbf{x}} \in \mathbb{R}^{n \times m}$$

In a neural network layer $\mathbf{y} = W\mathbf{x}$, the Jacobian tells you how each output dimension responds to each input dimension. Used in the chain rule during backprop.

**Row 5 — scalar / matrix:** loss $L$ differentiated w.r.t. a weight matrix $W \in \mathbb{R}^{m \times n}$.

$$\frac{\partial L}{\partial W} \in \mathbb{R}^{m \times n}, \qquad \left[\frac{\partial L}{\partial W}\right]_{ij} = \frac{\partial L}{\partial W_{ij}}$$

Entry $(i,j)$ says how much $L$ changes if you nudge weight $W_{ij}$. The gradient has exactly the same shape as $W$ — so `param.grad` in PyTorch always matches `param`.

#### Why this matters for backpropagation

Every layer in a neural network is a composition of functions. Backprop applies the chain rule repeatedly:

$$\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial \mathbf{y}}{\partial \mathbf{x}} \cdot \frac{\partial L}{\partial \mathbf{y}} = J \cdot \frac{\partial L}{\partial \mathbf{y}}$$

The Jacobian $J \in \mathbb{R}^{n \times m}$ routes gradients from output space back to input space. Shapes check out: $(n \times m)(m,) = (n,)$ ✓. For a linear layer $\mathbf{y} = W\mathbf{x}$, $J = W^\top$, so this becomes the familiar $W^\top \frac{\partial L}{\partial \mathbf{y}}$.

---

### 12.2 Scalar by Vector

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



---

**Example 3 — Quadratic form: $y = \mathbf{x}^\top A \mathbf{x}$**

$$y = \sum_i \sum_j x_i A_{ij} x_j$$

$$\frac{\partial y}{\partial \mathbf{x}} = (A + A^\top)\mathbf{x} = 2A\mathbf{x} \quad \text{(if } A \text{ symmetric)}$$

**Concrete numbers** ($A$ symmetric, $2 \times 2$):

$$A = \begin{bmatrix}2 & 1\\1 & 3\end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix}1\\2\end{bmatrix}$$

$$y = \begin{bmatrix}1&2\end{bmatrix}\begin{bmatrix}2&1\\1&3\end{bmatrix}\begin{bmatrix}1\\2\end{bmatrix} = \begin{bmatrix}1&2\end{bmatrix}\begin{bmatrix}4\\7\end{bmatrix} = 4+14 = 18$$

$$\frac{\partial y}{\partial \mathbf{x}} = 2A\mathbf{x} = 2\begin{bmatrix}2&1\\1&3\end{bmatrix}\begin{bmatrix}1\\2\end{bmatrix} = 2\begin{bmatrix}4\\7\end{bmatrix} = \begin{bmatrix}8\\14\end{bmatrix}$$

---

### 12.3 Scalar by Matrix

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



---

### 12.4 Vector by Vector — Jacobian

When both input and output are vectors, the derivative is a **Jacobian matrix**.

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = J \in \mathbb{R}^{n \times m}, \qquad J_{ij} = \frac{\partial y_j}{\partial x_i}$$

Row $i$ = input index, column $j$ = output index (denominator layout).

---

**Example 6 — Linear layer: $\mathbf{y} = W\mathbf{x}$, $W \in \mathbb{R}^{m \times n}$**

$$y_j = \sum_k W_{jk} x_k \quad \Rightarrow \quad \frac{\partial y_j}{\partial x_i} = W_{ji}$$

$$J_{ij} = W_{ji} \quad \Rightarrow \quad J = W^\top \in \mathbb{R}^{n \times m}$$

**Verify backprop:** $\dfrac{\partial \mathcal{L}}{\partial \mathbf{x}} = J \cdot \dfrac{\partial \mathcal{L}}{\partial \mathbf{y}} = W^\top \dfrac{\partial \mathcal{L}}{\partial \mathbf{y}}$ — the familiar backprop formula. ✓

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



---

### 12.5 Chain Rule in Matrix Form

Backpropagation is the chain rule applied repeatedly through layers. For the concept — composition of maps, computational graphs, and why reverse-mode is the efficient direction for a scalar loss — see [multivariable_calculus.md](multivariable_calculus.md) §4. This section is the matrix mechanics with worked numbers.

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



---

**Example 10 — Linear layer + ReLU + MSE (two-step backprop)**

$$\mathbf{z} = W\mathbf{x}, \quad \mathbf{h} = \text{ReLU}(\mathbf{z}), \quad \mathcal{L} = \|\mathbf{y} - \mathbf{h}\|_2^2$$



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

## 13. Least Squares

### 13.1 The Overdetermined Problem

When $A \in \mathbb{R}^{m \times n}$ has **more equations than unknowns** ($m > n$), $A\mathbf{x} = \mathbf{b}$
usually has **no exact solution** — $\mathbf{b}$ lies outside the column space $C(A)$ (§8.4). Instead, find the
$\mathbf{x}$ that comes closest:

$$\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \|A\mathbf{x} - \mathbf{b}\|_2^2$$

**Geometry:** $A\hat{\mathbf{x}}$ is the **orthogonal projection** of $\mathbf{b}$ onto $C(A)$ (§9.2). The
residual $\mathbf{r} = \mathbf{b} - A\hat{\mathbf{x}}$ is perpendicular to every column of $A$:

$$A^\top \mathbf{r} = \mathbf{0} \quad\Longleftrightarrow\quad A^\top(\mathbf{b} - A\hat{\mathbf{x}}) = \mathbf{0}$$

### 13.2 Normal Equations

Rearranging the orthogonality condition:

$$A^\top A\,\hat{\mathbf{x}} = A^\top \mathbf{b}$$

| Case | Solution |
|------|----------|
| $A$ full column rank ($A^\top A$ invertible, SPD) | $\hat{\mathbf{x}} = (A^\top A)^{-1} A^\top \mathbf{b} = A^{+}\mathbf{b}$ |
| Rank-deficient | Infinitely many minimizers; the **pseudoinverse** $A^{+}$ (from SVD, §7) picks the minimum-norm one |

$A^{+} = (A^\top A)^{-1}A^\top$ is the **Moore–Penrose pseudoinverse** (left inverse: $A^{+}A = I_n$). The
projection matrix onto $C(A)$ is $P = AA^{+} = A(A^\top A)^{-1}A^\top$ (§9.2).

**You can also derive the normal equations by calculus** (§11.1): set
$\nabla_{\mathbf{x}} \|A\mathbf{x}-\mathbf{b}\|_2^2 = 2A^\top(A\mathbf{x}-\mathbf{b}) = \mathbf{0}$.

> **Warning:** forming $A^\top A$ **squares the condition number** ($\kappa(A^\top A) = \kappa(A)^2$),
> so a mildly ill-conditioned $A$ gives a badly inaccurate $\hat{\mathbf{x}}$. Prefer QR (§13.3) or SVD.

### 13.3 Solving via QR

Factor $A = QR$ (reduced, §9.4). Because $Q$ has orthonormal columns, $\|A\mathbf{x} - \mathbf{b}\|_2$ is
minimized by

$$R\hat{\mathbf{x}} = Q^\top \mathbf{b} \qquad \text{(upper triangular — solve by back substitution)}$$

This never forms $A^\top A$, so it keeps the original conditioning. Substituting $A = QR$ into the normal
equations reproduces it exactly: $R^\top R \hat{\mathbf{x}} = R^\top Q^\top \mathbf{b} \Rightarrow R\hat{\mathbf{x}} = Q^\top \mathbf{b}$.

### 13.4 Polynomial Fitting & the Design Matrix

To fit $p_d(t) = c_0 + c_1 t + \cdots + c_d t^d$ to data points $(t_i, y_i)$, $i = 1, \ldots, m$, build the
**Vandermonde design matrix**:

$$A = \begin{bmatrix} 1 & t_1 & t_1^2 & \cdots & t_1^d \\ 1 & t_2 & t_2^2 & \cdots & t_2^d \\ \vdots & & & & \vdots \\ 1 & t_m & t_m^2 & \cdots & t_m^d \end{bmatrix} \in \mathbb{R}^{m \times (d+1)}, \qquad \mathbf{c} = \begin{bmatrix} c_0 \\ c_1 \\ \vdots \\ c_d \end{bmatrix}$$

Then $A\mathbf{c} \approx \mathbf{y}$ is an ordinary least-squares problem. The model is **linear in the
coefficients** $\mathbf{c}$ even though it is nonlinear in $t$ — this is why least squares applies. Any set of
basis functions $\phi_j(t)$ works the same way ($A_{ij} = \phi_j(t_i)$): Fourier terms, splines, radial basis
functions.

| Degree $d$ | Behavior |
|-----------|----------|
| Too low | **Underfits** — large residual, misses real structure |
| Well chosen | Small residual, smooth curve |
| Too high ($d \to m-1$) | **Overfits** — interpolates the noise; $A$ becomes ill-conditioned |

### 13.5 Regularization (Ridge / Tikhonov)

Add a penalty that keeps $\mathbf{x}$ small (or close to a prior $\mathbf{x}_0$):

$$\min_{\mathbf{x}} \|A\mathbf{x} - \mathbf{b}\|_2^2 + \lambda \|\mathbf{x} - \mathbf{x}_0\|_2^2, \qquad \lambda > 0$$

**Stacked form** — it is just ordinary least squares on a taller system:

$$A_{\text{aug}} = \begin{bmatrix} A \\ \sqrt{\lambda}\,I \end{bmatrix}, \qquad
\mathbf{b}_{\text{aug}} = \begin{bmatrix} \mathbf{b} \\ \sqrt{\lambda}\,\mathbf{x}_0 \end{bmatrix}
\quad\Longrightarrow\quad \min_{\mathbf{x}} \|A_{\text{aug}}\mathbf{x} - \mathbf{b}_{\text{aug}}\|_2^2$$

Closed form ($\mathbf{x}_0 = \mathbf{0}$):

$$\hat{\mathbf{x}}_{\text{ridge}} = (A^\top A + \lambda I)^{-1} A^\top \mathbf{b}$$

| Effect of $\lambda$ | |
|---|---|
| $A^\top A + \lambda I$ is **always** invertible ($\lambda > 0$) | Fixes rank-deficiency and ill-conditioning |
| Shrinks $\hat{\mathbf{x}}$ toward $\mathbf{x}_0$ | Trades a little bias for much less variance |
| $\lambda \to 0$ | Recovers ordinary least squares |
| $\lambda \to \infty$ | $\hat{\mathbf{x}} \to \mathbf{x}_0$ |

This is **ridge regression** / **weight decay**. Using an L1 penalty $\lambda\|\mathbf{x}\|_1$ instead gives
**Lasso** (sparse solutions, no closed form). **Weighted** least squares generalizes the data term to
$\|W^{1/2}(A\mathbf{x} - \mathbf{b})\|_2^2$ for a diagonal weight matrix $W$.

### 13.6 Equality-Constrained Least Squares (KKT)

Minimize the residual while forcing $\mathbf{x}$ to satisfy exact linear constraints:

$$\min_{\mathbf{x}} \|A\mathbf{x} - \mathbf{b}\|_2^2 \quad \text{subject to} \quad C\mathbf{x} = \mathbf{d}, \qquad C \in \mathbb{R}^{q \times n}$$

Form the **Lagrangian** $\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}) = \|A\mathbf{x} - \mathbf{b}\|_2^2 + 2\boldsymbol{\lambda}^\top(C\mathbf{x} - \mathbf{d})$
and set both gradients to zero. This gives the **KKT system** — one symmetric block matrix:

$$\begin{bmatrix} A^\top A & C^\top \\ C & 0 \end{bmatrix} \begin{bmatrix} \mathbf{x} \\ \boldsymbol{\lambda} \end{bmatrix} = \begin{bmatrix} A^\top \mathbf{b} \\ \mathbf{d} \end{bmatrix}$$

| Block row | Enforces |
|-----------|----------|
| Top: $A^\top A\,\mathbf{x} + C^\top \boldsymbol{\lambda} = A^\top \mathbf{b}$ | Stationarity (least-squares optimality along the constraint surface) |
| Bottom: $C\mathbf{x} = \mathbf{d}$ | Feasibility (the constraint itself) |

$\boldsymbol{\lambda}$ are the **Lagrange multipliers** — the sensitivity of the optimal cost to loosening
each constraint. Example: forcing coefficients to sum to one, $C = [1\ 1\ \cdots\ 1]$, $\mathbf{d} = 1$.

### 13.7 Reference Table

| Problem | Solution |
|---------|----------|
| $\min \|A\mathbf{x} - \mathbf{b}\|_2^2$ | $A^\top A\,\hat{\mathbf{x}} = A^\top \mathbf{b}$; or $R\hat{\mathbf{x}} = Q^\top \mathbf{b}$ |
| Ridge: $+\,\lambda\|\mathbf{x}\|_2^2$ | $(A^\top A + \lambda I)\hat{\mathbf{x}} = A^\top \mathbf{b}$ |
| Constrained: $C\mathbf{x} = \mathbf{d}$ | KKT block system above |
| Underdetermined ($m < n$), min-norm | $\hat{\mathbf{x}} = A^\top(AA^\top)^{-1}\mathbf{b}$ |
| Rank-deficient, min-norm | $\hat{\mathbf{x}} = A^{+}\mathbf{b}$ via SVD (§7) |

---

## 14. Linear Dynamical Systems

Eigenvalues (§6) exist to answer one question: **what does $A$ do when applied over and over?**

### 14.1 Discrete Systems & Matrix Powers

$$\mathbf{x}_{t+1} = A\mathbf{x}_t \quad\Longrightarrow\quad \mathbf{x}_t = A^t \mathbf{x}_0$$

Diagonalize $A = V\Lambda V^{-1}$ (§6.3). Then powers are trivial:

$$A^t = V \Lambda^t V^{-1} = V \operatorname{diag}(\lambda_1^t, \ldots, \lambda_n^t) V^{-1}$$

Expanding $\mathbf{x}_0 = \sum_i c_i \mathbf{v}_i$ in the eigenbasis ($\mathbf{c} = V^{-1}\mathbf{x}_0$):

$$\mathbf{x}_t = \sum_{i=1}^n c_i\, \lambda_i^t\, \mathbf{v}_i$$

Each **mode** $\mathbf{v}_i$ evolves independently, scaled by $\lambda_i^t$. The behavior is decided entirely by
the eigenvalues.

### 14.2 Stability & Steady States

Let the **spectral radius** be $\rho(A) = \max_i |\lambda_i|$.

| Condition | Long-run behavior of $\mathbf{x}_t$ |
|-----------|------------------------------------|
| $\rho(A) < 1$ | $\mathbf{x}_t \to \mathbf{0}$ — every mode decays |
| $\rho(A) > 1$ | $\|\mathbf{x}_t\| \to \infty$ — the mode(s) with largest $|\lambda|$ dominate and blow up |
| $\rho(A) = 1$, simple dominant $\lambda = 1$ | $\mathbf{x}_t \to c_1 \mathbf{v}_1$ — a nonzero **steady state** along the dominant eigenvector |
| $|\lambda| = 1$ complex | Undamped oscillation / rotation |

For large $t$ the term with the largest $|\lambda_i|$ swamps the rest — the **dominant eigenpair** governs the
asymptotics. Normalizing $\mathbf{x}_t/\|\mathbf{x}_t\|$ at each step to extract it is **power iteration**
(the basis of PageRank, §6.2).

**Gradient-descent connection.** One GD step $\mathbf{x} \leftarrow (I - \eta H)\mathbf{x}$ (on a quadratic with
Hessian $H$, §11.3) is a linear system with iteration matrix $I - \eta H$; it converges iff
$|1 - \eta\lambda_i| < 1$ for every eigenvalue $\lambda_i$ of $H$, i.e. $0 < \eta < 2/\lambda_{\max}$.

### 14.3 Markov Chains

A **stochastic matrix** $P$ has nonnegative entries with each column summing to $1$ (so probability is
conserved: $\mathbf{1}^\top P = \mathbf{1}^\top$). Then:

- $\lambda = 1$ is always an eigenvalue, and $\rho(P) = 1$ (Perron–Frobenius).
- If $P$ is irreducible and aperiodic, the dominant left eigenvector normalizes to the unique **stationary
  distribution** $\boldsymbol{\pi}$ with $P\boldsymbol{\pi} = \boldsymbol{\pi}$, and $\mathbf{x}_t \to \boldsymbol{\pi}$ from any start.

**Example — a doubly stochastic averaging step** (rows *and* columns sum to 1), e.g. each node replacing its
value by $\tfrac12$ itself $+\ \tfrac14$ each neighbor around a ring. Here $\boldsymbol{\pi}$ is uniform, so
every coordinate converges to the **average of the initial values** — the total is conserved and spreads out evenly.

### 14.4 Continuous Systems & the Matrix Exponential

The continuous-time analogue is the linear ODE

$$\dot{\mathbf{x}}(t) = A\mathbf{x}(t) \quad\Longrightarrow\quad \mathbf{x}(t) = e^{At}\,\mathbf{x}(0)$$

The **matrix exponential** is defined by the same series as the scalar one:

$$e^{At} = \sum_{k=0}^{\infty} \frac{(At)^k}{k!} = I + At + \frac{(At)^2}{2!} + \cdots$$

| Property | |
|----------|---|
| $e^{A \cdot 0} = I$ | |
| $\frac{d}{dt} e^{At} = A e^{At} = e^{At} A$ | why it solves the ODE |
| $e^{A(s+t)} = e^{As}e^{At}$ | but $e^{A+B} \neq e^A e^B$ unless $AB = BA$ |
| $\det(e^{A}) = e^{\operatorname{tr}(A)}$ | always invertible |

**Compute it by diagonalization** ($A = V\Lambda V^{-1}$):

$$e^{At} = V \operatorname{diag}(e^{\lambda_1 t}, \ldots, e^{\lambda_n t})\, V^{-1}$$

So $\mathbf{x}(t) = \sum_i c_i\, e^{\lambda_i t}\, \mathbf{v}_i$ — modes again, now with continuous growth/decay rates.

| Condition on eigenvalues of $A$ | Behavior of $\dot{\mathbf{x}} = A\mathbf{x}$ |
|-------------------------------|---------------------------------------------|
| $\operatorname{Re}(\lambda_i) < 0$ for all $i$ | Asymptotically stable — $\mathbf{x}(t) \to \mathbf{0}$ |
| Some $\operatorname{Re}(\lambda_i) > 0$ | Unstable — blows up |
| $\operatorname{Re}(\lambda_i) \leq 0$, a simple $\lambda = 0$ | Converges to a steady state in $N(A)$ |
| $\operatorname{Im}(\lambda_i) \neq 0$ | Oscillation at angular frequency $\operatorname{Im}(\lambda_i)$ |

(Contrast with the discrete test, which compares $|\lambda|$ to $1$; the continuous test compares
$\operatorname{Re}(\lambda)$ to $0$. The map $\lambda \mapsto e^{\lambda t}$ sends the left half-plane to the unit disk.)

### 14.5 The Graph Laplacian

Both the discrete averaging example and continuous diffusion are governed by the **graph Laplacian**
$L = D - W$, where $W$ is the (symmetric) edge-weight matrix and $D$ is the diagonal degree matrix. Equivalently
$L = A^\top A$ for the incidence matrix $A$ of §8.4.

| Property | Consequence |
|----------|-------------|
| Symmetric positive semidefinite | Real eigenvalues $0 = \lambda_1 \leq \lambda_2 \leq \cdots$ |
| Every row sums to $0$ ($L\mathbf{1} = \mathbf{0}$) | $\mathbf{1}$ is an eigenvector with eigenvalue $0$ |
| Multiplicity of eigenvalue $0$ | Number of connected components of the graph |
| $\lambda_2$ (**Fiedler value**) | Algebraic connectivity; its eigenvector bisects the graph (spectral clustering) |

**Diffusion** $\dot{\mathbf{x}} = -L\mathbf{x}$: eigenvalues of $-L$ are $0 \geq -\lambda_2 \geq \cdots$, so every
mode except the constant one decays. The $\lambda = 0$ mode ($\mathbf{1}$) is conserved — the total
$\sum_i x_i$ never changes — and $\mathbf{x}(t)$ relaxes to the **average** of the initial values, at a rate
set by $\lambda_2$. This is heat flow, consensus/averaging in multi-agent systems, and label propagation.

---

## 15. AI Applications

### 15.1 Neural Network Forward Pass

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



### 15.2 Attention Mechanism

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



### 15.3 PCA

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



**In AI:**
- Reduce input dimensionality before training
- Visualize high-dimensional embeddings (word vectors, latent spaces)
- Initialize weights or analyze learned representations

---

### 15.4 K-Means Clustering

K-means partitions $N$ points $\mathbf{x}_1, \ldots, \mathbf{x}_N \in \mathbb{R}^D$ into $K$ groups by
alternating two linear-algebra operations until the assignment stops changing.

**Objective** — minimize the total squared L2 distance from each point to its cluster centroid:

$$J = \sum_{i=1}^{N} \left\| \mathbf{x}_i - \boldsymbol{\mu}_{c_i} \right\|_2^2, \qquad c_i \in \{1, \ldots, K\}$$

| Term | Meaning |
|------|---------|
| $c_i$ | Cluster index assigned to point $\mathbf{x}_i$ |
| $\boldsymbol{\mu}_k \in \mathbb{R}^D$ | Centroid (mean vector) of cluster $k$ |
| $\|\mathbf{x}_i - \boldsymbol{\mu}_{c_i}\|_2^2$ | Squared Euclidean distance — the quantity §1.4 calls the L2 norm |

**The algorithm — Lloyd's iteration:**

| Step | Operation | Linear algebra |
|------|-----------|----------------|
| **0. Init** | Pick $K$ initial centroids (e.g. $K$ random points) | — |
| **1. Assign** | $c_i \leftarrow \arg\min_k \|\mathbf{x}_i - \boldsymbol{\mu}_k\|_2$ | Nearest-centroid by L2 distance |
| **2. Update** | $\boldsymbol{\mu}_k \leftarrow \dfrac{1}{|S_k|} \sum_{i \in S_k} \mathbf{x}_i$ | Feature-wise mean of the points in cluster $S_k$ |
| **3. Repeat** | Alternate steps 1–2 until assignments stabilize | $J$ decreases monotonically → converges |

**Why the centroid is the mean:** for a fixed set $S_k$, the point $\boldsymbol{\mu}$ minimizing
$\sum_{i \in S_k} \|\mathbf{x}_i - \boldsymbol{\mu}\|_2^2$ is found by setting the gradient to zero
(§11.1, $\nabla_{\boldsymbol{\mu}} \|\mathbf{x}_i - \boldsymbol{\mu}\|_2^2 = -2(\mathbf{x}_i - \boldsymbol{\mu})$):

$$\sum_{i \in S_k} -2(\mathbf{x}_i - \boldsymbol{\mu}) = \mathbf{0} \quad\Longrightarrow\quad \boldsymbol{\mu} = \frac{1}{|S_k|}\sum_{i \in S_k} \mathbf{x}_i$$

**Vectorized distance to all centroids.** Stack centroids as rows of $M \in \mathbb{R}^{K \times D}$. Broadcasting
$X \in \mathbb{R}^{N \times D}$ against $M$ gives an $(N \times K)$ distance matrix in one shot:

$$\text{dist}_2^2(X, M)_{ik} = \|\mathbf{x}_i\|_2^2 - 2\,\mathbf{x}_i^\top \boldsymbol{\mu}_k + \|\boldsymbol{\mu}_k\|_2^2$$

The cross term $X M^\top \in \mathbb{R}^{N \times K}$ is a single matrix multiply (§2.3); the two norm terms are
per-row / per-column vectors added by broadcasting. `argmin` along axis 1 gives the assignments.

**Concrete example** ($D=1$, $K=2$, points $\{1, 2, 10, 12\}$, init $\mu_1=1,\ \mu_2=2$):

| Iter | Assign ($c$) | Update |
|------|-------------|--------|
| 1 | $1\!\to\!\mu_1;\ 2,10,12\!\to\!\mu_2$ | $\mu_1 = 1,\quad \mu_2 = (2+10+12)/3 = 8$ |
| 2 | $1,2\!\to\!\mu_1;\ 10,12\!\to\!\mu_2$ | $\mu_1 = 1.5,\quad \mu_2 = 11$ |
| 3 | $1,2\!\to\!\mu_1;\ 10,12\!\to\!\mu_2$ | unchanged → **converged** |

Final $J = (0.5^2 + 0.5^2) + (1^2 + 1^2) = 2.5$.

**In AI:** vector quantization, image color compression, feature learning (bag-of-visual-words),
initializing Gaussian mixture models, and building the codebooks used in some tokenizers (e.g. VQ-VAE).

> **Caveats:** K-means assumes roughly spherical, equal-size clusters (it only sees L2 distance); the result
> depends on initialization (run several times, keep the lowest $J$, or use k-means++); $K$ must be chosen in advance.

---

### 15.5 K-Nearest Neighbors (KNN)

KNN is a non-parametric classifier: there is no training beyond **storing** the labeled set. A test point is
labeled by a majority vote of its $k$ closest training points under the L2 metric.

| Step | Operation |
|------|-----------|
| **Train** | Store $X_{\text{train}} \in \mathbb{R}^{M \times D}$ and labels $\mathbf{y} \in \{0,\ldots,C-1\}^M$ |
| **Distance** | For each test point, L2 distance to every training point |
| **Vote** | Take the $k$ smallest distances; predict the most common label among them |

**Pairwise distance matrix.** With $X_{\text{test}} \in \mathbb{R}^{N \times D}$, broadcasting a
$(N, 1, D)$ array against a $(1, M, D)$ array produces the $(N, M, D)$ difference tensor; reducing the last
axis with an L2 norm gives

$$\Big[\text{dist}(X_{\text{test}}, X_{\text{train}})\Big]_{ij} = \left\| \mathbf{x}^{\text{test}}_i - \mathbf{x}^{\text{train}}_j \right\|_2 \in \mathbb{R}^{N \times M}$$

Equivalently, and faster, via the same expansion as §15.4:

$$D^2 = \|X_{\text{test}}\|^2_{\text{row}} \; - \; 2\,X_{\text{test}} X_{\text{train}}^\top \; + \; \|X_{\text{train}}\|^2_{\text{col}}$$

where $X_{\text{test}} X_{\text{train}}^\top \in \mathbb{R}^{N \times M}$ is one matrix multiply and the norm
terms broadcast over rows and columns.

**Prediction:** `idx = argsort(D, axis=1)[:, :k]` selects the $k$ nearest training indices per test row;
`y_pred[i] = mode(y_train[idx[i]])`.

| Hyperparameter | Effect |
|----------------|--------|
| Small $k$ (e.g. 1) | Low bias, high variance — sensitive to noise / outliers |
| Large $k$ | Smoother decision boundary, higher bias; too large washes out real structure |
| Distance metric | L2 is standard; features should be scaled first so no dimension dominates the norm |

**Concrete example** ($k=3$): test point $\mathbf{x}$ has the 3 nearest training labels $\{A, A, B\}$
→ predict $A$ (2 votes vs 1).

**In AI:** strong baseline for classification, k-NN retrieval over embedding vectors
(semantic search, RAG, face recognition, recommendation), and label propagation. At scale the exact
$X X^\top$ distance computation is replaced by approximate nearest-neighbor indexes (FAISS, HNSW), but the
underlying quantity is still the L2 (or cosine, §1.3) distance between vectors.

> **Cost:** no training time, but inference is $O(NMD)$ — every prediction scans the whole training set.
> Memory grows linearly with the data. This is the opposite trade-off from a neural network.
