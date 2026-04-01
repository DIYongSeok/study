# NumPy

NumPy (Numerical Python) is the foundational library for scientific computing in Python. It provides a high-performance N-dimensional array object and tools for mathematical operations.

```bash
pip install numpy
```

```python
import numpy as np
```

## Table of Contents

1. [Array Creation](#1-array-creation)
   - 1.1 [From Python Data](#11-from-python-data)
   - 1.2 [Built-in Constructors](#12-built-in-constructors)
   - 1.3 [Array Attributes](#13-array-attributes)
2. [Indexing & Slicing](#2-indexing--slicing)
   - 2.1 [1D Indexing](#21-1d-indexing)
   - 2.2 [2D Indexing](#22-2d-indexing)
   - 2.3 [Boolean Indexing](#23-boolean-indexing)
   - 2.4 [Fancy Indexing](#24-fancy-indexing)
3. [Array Manipulation](#3-array-manipulation)
   - 3.1 [Shape & Reshape](#31-shape--reshape)
   - 3.2 [Stacking & Splitting](#32-stacking--splitting)
   - 3.3 [Sorting & Searching](#33-sorting--searching)
4. [Math Operations](#4-math-operations)
   - 4.1 [Element-wise Arithmetic](#41-element-wise-arithmetic)
   - 4.2 [Broadcasting](#42-broadcasting)
   - 4.3 [Aggregation](#43-aggregation)
   - 4.4 [Universal Functions (ufunc)](#44-universal-functions-ufunc)
5. [Linear Algebra](#5-linear-algebra)
6. [Random Module](#6-random-module)
7. [NumPy vs Python List](#7-numpy-vs-python-list)

---

## 1. Array Creation

### 1.1 From Python Data

```python
# 1D array from list
a = np.array([1, 2, 3, 4, 5])

# 2D array (matrix) from list of lists
m = np.array([[1, 2, 3],
              [4, 5, 6]])

# Specify dtype explicitly
f = np.array([1, 2, 3], dtype=np.float32)
i = np.array([1.7, 2.9], dtype=np.int32)  # truncates to [1, 2]
```

### 1.2 Built-in Constructors

```python
np.zeros((3, 4))          # 3×4 matrix of 0.0
np.ones((2, 3))           # 2×3 matrix of 1.0
np.full((2, 3), 7)        # 2×3 matrix filled with 7
np.eye(3)                 # 3×3 identity matrix

np.arange(0, 10, 2)       # [0 2 4 6 8]  — like range()
np.arange(5)              # [0 1 2 3 4]

np.linspace(0, 1, 5)      # [0.   0.25 0.5  0.75 1.  ]  — n evenly spaced points
np.logspace(0, 3, 4)      # [1. 10. 100. 1000.]  — log scale

np.empty((2, 2))          # uninitialized (values are whatever was in memory)
```

### 1.3 Array Attributes

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a.shape)    # (2, 3)   — (rows, columns)
print(a.ndim)     # 2        — number of dimensions
print(a.size)     # 6        — total number of elements
print(a.dtype)    # int64    — element data type
print(a.itemsize) # 8        — bytes per element
```

---

## 2. Indexing & Slicing

### 2.1 1D Indexing

```python
a = np.array([10, 20, 30, 40, 50])

print(a[0])       # 10   — first element
print(a[-1])      # 50   — last element
print(a[1:4])     # [20 30 40]  — slice [start:stop)
print(a[:3])      # [10 20 30]
print(a[2:])      # [30 40 50]
print(a[::2])     # [10 30 50]  — every 2nd element
print(a[::-1])    # [50 40 30 20 10]  — reversed
```

### 2.2 2D Indexing

```python
m = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(m[0, 0])      # 1    — row 0, col 0
print(m[1, 2])      # 6    — row 1, col 2
print(m[0, :])      # [1 2 3]  — entire row 0
print(m[:, 1])      # [2 5 8]  — entire column 1
print(m[0:2, 1:3])  # [[2 3] [5 6]]  — submatrix

# Modifying a submatrix
m[0, :] = 0         # set entire row 0 to 0
```

### 2.3 Boolean Indexing

Filter elements by a condition — returns a 1D array of matching values.

```python
a = np.array([10, 20, 30, 40, 50])

mask = a > 25
print(mask)       # [False False  True  True  True]
print(a[mask])    # [30 40 50]

# Inline
print(a[a % 20 == 0])    # [20 40]

# Modify elements meeting a condition
a[a < 30] = 0
print(a)          # [0 0 30 40 50]
```

### 2.4 Fancy Indexing

Select elements using an array of indices.

```python
a = np.array([10, 20, 30, 40, 50])

idx = [0, 2, 4]
print(a[idx])     # [10 30 50]

# 2D fancy indexing
m = np.array([[1, 2], [3, 4], [5, 6]])
rows = [0, 2]
print(m[rows])    # [[1 2] [5 6]]
```

---

## 3. Array Manipulation

### 3.1 Shape & Reshape

```python
a = np.arange(12)          # [0 1 2 ... 11]

b = a.reshape(3, 4)        # 3 rows, 4 columns (must be compatible size)
c = a.reshape(2, 2, 3)     # 3D array

# -1 lets NumPy infer the missing dimension
d = a.reshape(3, -1)       # same as reshape(3, 4)

# Flatten back to 1D
print(b.flatten())         # always returns a copy
print(b.ravel())           # returns a view if possible (faster)

# Add/remove dimensions
e = a.reshape(1, 12)       # shape (1, 12)
f = np.expand_dims(a, axis=0)  # same as above
g = np.squeeze(e)          # remove dimensions of size 1 → shape (12,)

# Transpose
m = np.array([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
print(m.T)                 # shape (3, 2)
```

### 3.2 Stacking & Splitting

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Stacking
np.vstack([a, b])          # [[1 2 3] [4 5 6]]  — vertical (row-wise)
np.hstack([a, b])          # [1 2 3 4 5 6]      — horizontal (column-wise)
np.concatenate([a, b])     # [1 2 3 4 5 6]      — general, specify axis

A = np.ones((2, 3))
B = np.ones((2, 3))
np.vstack([A, B])          # shape (4, 3)
np.hstack([A, B])          # shape (2, 6)

# Splitting
x = np.arange(9)
np.split(x, 3)             # [array([0,1,2]), array([3,4,5]), array([6,7,8])]
np.array_split(x, 4)       # allows uneven splits
```

### 3.3 Sorting & Searching

```python
a = np.array([3, 1, 4, 1, 5, 9, 2])

np.sort(a)                 # [1 1 2 3 4 5 9]  — returns sorted copy
a.sort()                   # sorts in-place

np.argsort(a)              # indices that would sort the array

np.argmax(a)               # index of maximum value
np.argmin(a)               # index of minimum value

np.where(a > 3)            # indices where condition is True
np.where(a > 3, 1, 0)      # replace: 1 where True, 0 where False
```

---

## 4. Math Operations

### 4.1 Element-wise Arithmetic

All standard operators work element-wise. Arrays must have compatible shapes.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)     # [5 7 9]
print(a - b)     # [-3 -3 -3]
print(a * b)     # [4 10 18]
print(a / b)     # [0.25 0.4  0.5]
print(a // b)    # [0 0 0]   floor division
print(a % b)     # [1 2 3]   modulo
print(a ** 2)    # [1 4 9]   exponentiation
```

### 4.2 Broadcasting

Broadcasting allows operations between arrays of different shapes by automatically expanding the smaller one.

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])   # shape (2, 3)

# Scalar broadcast — applied to every element
print(a + 10)              # [[11 12 13] [14 15 16]]
print(a * 2)               # [[2 4 6] [8 10 12]]

# 1D array broadcast across rows
row = np.array([10, 20, 30])   # shape (3,) → treated as (1, 3)
print(a + row)
# [[11 22 33]
#  [14 25 36]]

# Column vector broadcast across columns
col = np.array([[10], [20]])   # shape (2, 1)
print(a + col)
# [[11 12 13]
#  [24 25 26]]
```

Broadcasting rules:
1. If arrays differ in number of dimensions, prepend 1s to the smaller shape.
2. Sizes must match or one of them must be 1, in each dimension.

### 4.3 Aggregation

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

# Global
print(a.sum())     # 21
print(a.mean())    # 3.5
print(a.max())     # 6
print(a.min())     # 1
print(a.std())     # standard deviation
print(a.var())     # variance

# Along an axis
print(a.sum(axis=0))   # [5 7 9]   — sum each column (collapse rows)
print(a.sum(axis=1))   # [6 15]    — sum each row (collapse columns)
print(a.max(axis=1))   # [3 6]
print(a.mean(axis=0))  # [2.5 3.5 4.5]

# Cumulative
print(np.cumsum(a))    # [1 3 6 10 15 21]  — running total
```

### 4.4 Universal Functions (ufunc)

Element-wise math functions.

```python
a = np.array([0, np.pi/2, np.pi])

np.sin(a)            # [0.  1.  0.]
np.cos(a)            # [1.  0. -1.]
np.exp(np.array([0, 1, 2]))   # [1.    2.718 7.389]
np.log(np.array([1, np.e, np.e**2]))  # [0. 1. 2.]
np.sqrt(np.array([1, 4, 9]))  # [1. 2. 3.]
np.abs(np.array([-3, -1, 2])) # [3 1 2]
np.round(np.array([1.5, 2.4, 3.7]))  # [2. 2. 4.]
```

---

## 5. Linear Algebra

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# Matrix multiplication
print(A @ B)              # [[19 22] [43 50]]
print(np.matmul(A, B))    # same as @

# Element-wise multiplication (NOT matrix multiply)
print(A * B)              # [[5 12] [21 32]]

# Transpose
print(A.T)                # [[1 3] [2 4]]

# Determinant
print(np.linalg.det(A))   # -2.0

# Inverse
print(np.linalg.inv(A))   # [[-2.   1. ] [ 1.5 -0.5]]

# Eigenvalues and eigenvectors
vals, vecs = np.linalg.eig(A)
print(vals)               # [-0.372 5.372]
print(vecs)               # columns are eigenvectors

# Solve linear system  Ax = b
b = np.array([5, 11])
x = np.linalg.solve(A, b)
print(x)                  # [1. 2.]  → A @ x == b

# Singular Value Decomposition
U, S, Vt = np.linalg.svd(A)
```

---

## 6. Random Module

```python
rng = np.random.default_rng(seed=42)   # preferred: reproducible random generator

# Uniform floats in [0, 1)
rng.random((2, 3))

# Random integers
rng.integers(0, 10, size=(3, 3))       # integers in [0, 10)

# Normal distribution  N(mean, std)
rng.normal(loc=0, scale=1, size=(4, 4))

# Shuffle and choice
a = np.arange(10)
rng.shuffle(a)                          # in-place
sample = rng.choice(a, size=5, replace=False)  # without replacement

# Older API (still common)
np.random.seed(42)
np.random.rand(3, 3)       # uniform [0, 1)
np.random.randn(3, 3)      # standard normal
np.random.randint(0, 10, size=(3, 3))
```

---

## 7. NumPy vs Python List

| Feature                  | Python List              | NumPy Array                     |
|--------------------------|--------------------------|---------------------------------|
| Type                     | Mixed types allowed      | Single dtype (homogeneous)      |
| `a * 2`                  | Repeats the list         | Multiplies each element by 2    |
| `a + b`                  | Concatenates             | Element-wise addition           |
| Math operations          | Need explicit loop       | Vectorized, no loop needed      |
| Multi-dimensional        | List of lists            | True N-D array with `.reshape`  |
| Memory                   | Each element is an object | Contiguous block (much smaller) |
| Speed (large data)       | Slow                     | ~10–100× faster                 |
| Broadcasting             | Not supported            | Built-in                        |

```python
import time

n = 10_000_000

# Python list
a = list(range(n))
t0 = time.time()
b = [x * 2 for x in a]
print(f"list: {time.time() - t0:.3f}s")

# NumPy
a = np.arange(n)
t0 = time.time()
b = a * 2
print(f"numpy: {time.time() - t0:.3f}s")
# numpy is typically 10–50× faster
```
