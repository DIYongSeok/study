# Python Advanced

## Table of Contents

1. [Concise Syntax](#1-concise-syntax)
   - 1.1 [Short Conditionals](#11-short-conditionals)
   - 1.2 [Short Loops](#12-short-loops)
   - 1.3 [Type Hints](#13-type-hints)
2. [Flexible Function Arguments](#2-flexible-function-arguments)
   - 2.1 [`*args` — Variable Positional Arguments](#21-args--variable-positional-arguments)
   - 2.2 [`**kwargs` — Variable Keyword Arguments](#22-kwargs--variable-keyword-arguments)
3. [Lambda & Functional Tools](#3-lambda--functional-tools)
   - 3.1 [Lambda Functions](#31-lambda-functions)
   - 3.2 [`map`, `filter`, `sorted`](#32-map-filter-sorted)
4. [Generators](#4-generators)
5. [Decorators](#5-decorators)
6. [Magic Methods (Dunder Methods)](#6-magic-methods-dunder-methods)
   - 6.1 [Common Magic Methods](#61-common-magic-methods)
   - 6.2 [Operator Overloading](#62-operator-overloading)
7. [Context Managers (`with`)](#7-context-managers-with)

---

## 1. Concise Syntax

### 1.1 Short Conditionals

When both branches are single statements, Python allows more compact forms.

```python
# Standard form
if condition:
    statement_a
else:
    statement_b

# Compact one-liner (both branches on the same line)
if condition: statement_a
else: statement_b

# Ternary expression (inline if-else)
result = statement_a if condition else statement_b

# Examples
x = 7
label = "odd" if x % 2 != 0 else "even"
print("positive" if x > 0 else "non-positive")
```

### 1.2 Short Loops

```python
# Standard form
result = []
for i in range(100):
    if i % 2 == 0:
        result.append(2 * i)

# List comprehension — one line
result = [2 * i for i in range(100) if i % 2 == 0]

# General syntax
# [expression  for item in iterable  if condition]
squares    = [x**2 for x in range(10)]
even_only  = [x for x in range(20) if x % 2 == 0]
```

### 1.3 Type Hints

Type hints declare expected types for variables, parameters, and return values. They do **not** enforce types at runtime — Python remains dynamically typed. Their value is readability and IDE/static-analysis support.

```python
# Variable annotation
count: int = 0
name: str = "Alice"

# Function with type hints
def add(a: int, b: int) -> int:
    return a + b

def greet(name: str, age: int) -> str:
    return f"Hello, {name}. You are {age} years old."

# Without hints — caller has no idea what types to pass
def process(data):
    ...

# With hints — intent is clear
def process(data: list[int]) -> dict[str, int]:
    ...
```

Complex types from the `typing` module (Python 3.8 and earlier):

```python
from typing import List, Dict, Tuple, Optional, Union

def scores(data: List[int]) -> Optional[float]:
    if not data:
        return None       # Optional means the return can be None
    return sum(data) / len(data)

def lookup(key: str) -> Union[int, str]:
    ...                   # Union means the return is int OR str
```

From Python 3.9+, built-in types can be used directly:

```python
def process(data: list[int]) -> dict[str, int]: ...
```

---

## 2. Flexible Function Arguments

### 2.1 `*args` — Variable Positional Arguments

Collects any number of positional arguments into a **tuple**.

```python
def add_many(*args):
    # args is a tuple: (1, 2, 3, ...)
    total = 0
    for num in args:
        total += num
    return total

print(add_many(1, 2, 3))         # 6
print(add_many(10, 20, 30, 40))  # 100

# Mixing with regular parameters
def first_and_rest(first, *rest):
    print("first:", first)
    print("rest:", rest)

first_and_rest(1, 2, 3, 4)
# first: 1
# rest: (2, 3, 4)
```

### 2.2 `**kwargs` — Variable Keyword Arguments

Collects any number of keyword arguments into a **dictionary**.

```python
def introduce(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

introduce(name="Alice", age=30, city="Seoul")
# name: Alice
# age: 30
# city: Seoul

# Combining all three parameter types
def mixed(a, b=10, *args, **kwargs):
    print(a, b, args, kwargs)

mixed(1, 2, 3, 4, x=5, y=6)
# 1 2 (3, 4) {'x': 5, 'y': 6}
```

---

## 3. Lambda & Functional Tools

### 3.1 Lambda Functions

An anonymous single-expression function. Used for short, one-time operations.

```python
# Syntax: lambda parameters: expression

# Standard def
def add(x, y):
    return x + y

# Equivalent lambda
add = lambda x, y: x + y
print(add(3, 5))  # 8

# Commonly used inline (not assigned to a variable)
result = (lambda x: x ** 2)(4)  # 16
```

### 3.2 `map`, `filter`, `sorted`

```python
nums = [1, 2, 3, 4, 5]

# map(function, iterable) — apply function to every element
squared = list(map(lambda x: x**2, nums))
print(squared)   # [1, 4, 9, 16, 25]

doubled = list(map(lambda x: x * 2, nums))
print(doubled)   # [2, 4, 6, 8, 10]

# filter(function, iterable) — keep elements where function returns True
evens = list(filter(lambda x: x % 2 == 0, nums))
print(evens)     # [2, 4]

# sorted(iterable, key=..., reverse=...) — sort by custom key
words = ["banana", "apple", "cherry", "date"]
print(sorted(words))                          # alphabetical
print(sorted(words, key=len))                 # by length
print(sorted(words, key=lambda w: w[-1]))     # by last character
print(sorted(nums, reverse=True))             # descending

# Sorting a list of dicts
students = [{"name": "Bob", "score": 85},
            {"name": "Alice", "score": 92},
            {"name": "Carol", "score": 78}]
by_score = sorted(students, key=lambda s: s["score"], reverse=True)
```

---

## 4. Generators

A generator **produces items one at a time** only when requested, instead of building the entire sequence in memory. Essential when dealing with large or infinite sequences.

```python
# Using yield instead of return
def countdown(n):
    while n > 0:
        yield n      # pause here, return n, resume on next call
        n -= 1

gen = countdown(3)
print(next(gen))  # 3
print(next(gen))  # 2
print(next(gen))  # 1
# next(gen) would raise StopIteration

# Use in a for loop (most common)
for val in countdown(5):
    print(val)     # 5 4 3 2 1

# Generator expression — like list comprehension but lazy
big = (x**2 for x in range(10**8))  # no memory used yet
print(next(big))  # 0
print(next(big))  # 1
```

Comparison with a list:

| Feature      | List `[...]`                | Generator `(...)` / `yield` |
|--------------|-----------------------------|-----------------------------|
| Memory       | All items stored at once    | One item at a time          |
| Speed        | Fast for repeated access    | Fast for single-pass        |
| Reusable     | Yes                         | No — exhausted after one pass |
| Best for     | Small data, random access   | Large/infinite sequences    |

---

## 5. Decorators

A decorator **wraps a function** to extend or modify its behavior without changing its code. Uses the `@` syntax.

```python
# Step 1 — define the decorator (a function that takes a function)
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("before")
        result = func(*args, **kwargs)  # call the original function
        print("after")
        return result
    return wrapper

# Step 2 — apply with @
@my_decorator
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")
# before
# Hello, Alice!
# after
```

`@my_decorator` is exactly equivalent to writing `greet = my_decorator(greet)`.

**Practical example — measuring execution time:**

```python
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f}s")
        return result
    return wrapper

@timer
def heavy_work():
    total = 0
    for i in range(1_000_000):
        total += i
    return total

heavy_work()
# heavy_work took 0.0523s
```

**Stacking decorators:**

```python
@decorator_a
@decorator_b
def func(): ...
# Applied bottom-up: decorator_a(decorator_b(func))
```

---

## 6. Magic Methods (Dunder Methods)

Methods named with double underscores (`__method__`) let your custom classes behave like built-in Python types. They are called automatically by the interpreter in certain situations.

### 6.1 Common Magic Methods

```python
class SmartCounter:
    def __init__(self, start=0):       # called on object creation
        self.count = start

    def __str__(self):                 # called by print() and str()
        return f"Counter({self.count})"

    def __repr__(self):                # called in the REPL / debugging
        return f"SmartCounter(start={self.count})"

    def __len__(self):                 # called by len()
        return self.count

    def __call__(self, increment=1):   # makes the object callable like a function
        self.count += increment
        print(f"count is now {self.count}")

c = SmartCounter(10)
print(c)       # Counter(10)        — __str__
print(len(c))  # 10                 — __len__
c(5)           # count is now 15   — __call__
c(2)           # count is now 17   — __call__
```

`__call__` is useful for stateful callables — the object remembers previous values across calls. This pattern is used heavily in decorators and machine-learning model layers.

### 6.2 Operator Overloading

Teach your class how to respond to operators like `+`, `-`, `==`.

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):          # v1 + v2
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):          # v1 - v2
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):         # v * 3
        return Vector(self.x * scalar, self.y * scalar)

    def __eq__(self, other):           # v1 == v2
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(1, 2)
v2 = Vector(3, 4)
print(v1 + v2)   # Vector(4, 6)
print(v1 * 3)    # Vector(3, 6)
print(v1 == v2)  # False
```

Common dunder methods for operators:

| Method       | Operator | Method       | Operator |
|--------------|----------|--------------|----------|
| `__add__`    | `+`      | `__eq__`     | `==`     |
| `__sub__`    | `-`      | `__lt__`     | `<`      |
| `__mul__`    | `*`      | `__gt__`     | `>`      |
| `__truediv__`| `/`      | `__len__`    | `len()`  |
| `__mod__`    | `%`      | `__contains__`| `in`   |

---

## 7. Context Managers (`with`)

A context manager wraps a block of code to guarantee **setup and cleanup** happen correctly — even if an exception is raised inside the block.

```python
# Without context manager — risky
f = open("data.txt", "w")
f.write("hello")
# if an error occurs here, f.close() never runs → file stays locked
f.close()

# With context manager — safe
with open("data.txt", "w") as f:
    f.write("hello")
# file is closed automatically here, even if an exception occurred
```

**How it works** — two magic methods:

- `__enter__`: runs when entering the `with` block; returns the resource.
- `__exit__`: runs when leaving (even on exception); handles cleanup.

**Creating a custom context manager:**

```python
class ManagedResource:
    def __enter__(self):
        print("acquiring resource")
        return self       # value bound to the 'as' variable

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("releasing resource")
        return False      # False = don't suppress exceptions

with ManagedResource() as res:
    print("using resource")
# acquiring resource
# using resource
# releasing resource
```

**Using `contextlib` for a simpler approach:**

```python
from contextlib import contextmanager

@contextmanager
def managed():
    print("setup")
    yield              # code inside 'with' block runs here
    print("teardown")

with managed():
    print("working")
# setup
# working
# teardown
```

