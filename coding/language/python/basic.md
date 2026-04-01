# Python Basic

## Table of Contents

1. [Program Structure](#1-program-structure)
   - 1.1 [Hello World & Output](#11-hello-world--output)
   - 1.2 [User Input](#12-user-input)
2. [Data Types](#2-data-types)
   - 2.1 [Numeric](#21-numeric)
   - 2.2 [String](#22-string)
   - 2.3 [List](#23-list)
   - 2.4 [Tuple](#24-tuple)
   - 2.5 [Dictionary](#25-dictionary)
   - 2.6 [Type Conversion](#26-type-conversion)
3. [Control Flow](#3-control-flow)
   - 3.1 [Conditionals (`if` / `elif` / `else`)](#31-conditionals-if--elif--else)
   - 3.2 [While Loop](#32-while-loop)
   - 3.3 [For Loop & `range`](#33-for-loop--range)
   - 3.4 [`break` and `continue`](#34-break-and-continue)
4. [Functions](#4-functions)
   - 4.1 [Defining Functions](#41-defining-functions)
   - 4.2 [Return vs Print](#42-return-vs-print)
   - 4.3 [Default Parameters](#43-default-parameters)
   - 4.4 [Scope — Local vs Global](#44-scope--local-vs-global)
   - 4.5 [Recursion](#45-recursion)
5. [Comprehensions](#5-comprehensions)
   - 5.1 [List Comprehension](#51-list-comprehension)
   - 5.2 [Dictionary Comprehension](#52-dictionary-comprehension)
6. [Object-Oriented Programming](#6-object-oriented-programming)
   - 6.1 [Class Basics](#61-class-basics)
   - 6.2 [Inheritance & Overriding](#62-inheritance--overriding)
7. [Modules](#7-modules)
   - 7.1 [Creating and Importing a Module](#71-creating-and-importing-a-module)
   - 7.2 [Module in a Folder (Package)](#72-module-in-a-folder-package)
8. [Exception Handling](#8-exception-handling)

---

## 1. Program Structure

### 1.1 Hello World & Output

```python
print("Hello, World!")
print(1 + 2)           # 3
print("a", "b", "c")  # a b c   (space-separated by default)
print("a", "b", sep="-")  # a-b
print("line1", end=" ")   # suppress newline
print("line2")            # line1 line2
```

### 1.2 User Input

`input()` always returns a **string**. Convert it explicitly when you need a number.

```python
# Single value
x = int(input())        # read one integer

# Multiple values on one line
line = input()          # e.g. "10 20"
parts = line.split()    # ["10", "20"]
a, b = int(parts[0]), int(parts[1])
print(a + b)            # 30

# One-liner shorthand (common in competitive programming)
a, b = map(int, input().split())
print(a + b)
```

---

## 2. Data Types

### 2.1 Numeric

```python
# Integer
x = 10
# Float
y = 3.14
# Operators
print(3 + 3)    # 6    addition
print(10 - 4)   # 6    subtraction
print(3 * 4)    # 12   multiplication
print(7 / 2)    # 3.5  true division (always float)
print(7 // 2)   # 3    floor division (integer result)
print(7 % 2)    # 1    modulo (remainder)
print(2 ** 10)  # 1024 exponentiation
```

### 2.2 String

```python
s1 = "hello"
s2 = 'world'

# Concatenation
print(s1 + " " + s2)   # hello world

# Repetition
print("ab" * 3)         # ababab

# f-string (preferred formatting)
name, score = "Alice", 95
print(f"{name} scored {score}")   # Alice scored 95

# Useful methods
print("hello world".split())      # ['hello', 'world']
print("hello world".split("o"))   # ['hell', ' w', 'rld']
print("  hello  ".strip())        # 'hello'
print("hello".upper())            # 'HELLO'
print("HELLO".lower())            # 'hello'
print("hello".replace("l", "r"))  # 'herro'
print("hello".find("ll"))         # 2  (index of first match, -1 if not found)
print(len("hello"))               # 5

# Indexing and slicing (same rules as list)
s = "python"
print(s[0])     # p
print(s[-1])    # n
print(s[1:4])   # yth
print(s[::-1])  # nohtyp  (reversed)
```

### 2.3 List

An ordered, mutable sequence.

```python
a = [1, 2, 3, 4, 5]

# Indexing
print(a[0])     # 1
print(a[-1])    # 5  (last element)

# Slicing  [start : stop : step]  (stop is exclusive)
print(a[1:3])   # [2, 3]
print(a[:3])    # [1, 2, 3]
print(a[2:])    # [3, 4, 5]
print(a[:])     # [1, 2, 3, 4, 5]  (copy)
print(a[::-1])  # [5, 4, 3, 2, 1]  (reversed)

# Common methods
print(len(a))      # 5
a.append(6)        # [1, 2, 3, 4, 5, 6]
a.pop()            # removes and returns last element → 6
a.pop(0)           # removes element at index 0 → 1
a.insert(1, 99)    # insert 99 at index 1
a.remove(99)       # remove first occurrence of 99
a.sort()           # sort in-place ascending
a.sort(reverse=True)  # sort descending
a.reverse()        # reverse in-place
print(a.index(3))  # index of first occurrence of 3
print(3 in a)      # True / False membership test

# 2D list (matrix)
matrix = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]
print(matrix[0][0])  # 1
print(matrix[1][2])  # 6
print(matrix[2][2])  # 9
```

### 2.4 Tuple

Ordered and **immutable** — cannot be changed after creation.

```python
t = (1, 2, 3)
print(t[0])     # 1
# t[0] = 10    # TypeError — tuples are immutable

# Useful for multiple return values
def min_max(lst):
    return min(lst), max(lst)

lo, hi = min_max([3, 1, 4, 1, 5])
print(lo, hi)   # 1 5
```

### 2.5 Dictionary

An unordered collection of **key-value pairs**.

```python
student = {"name": "Alice", "score": 90}

# Access
print(student["name"])          # Alice
print(student.get("grade", "N/A"))  # N/A (safe access with default)

# Modify
student["score"] = 95           # update existing key
student["grade"] = "A"          # add new key

# Delete
del student["grade"]

# Iteration
for key, value in student.items():
    print(f"{key}: {value}")

# Useful methods
print(student.keys())           # dict_keys(['name', 'score'])
print(student.values())         # dict_values(['Alice', 95])
print("name" in student)        # True
```

### 2.6 Type Conversion

```python
int("3")        # 3
float("3.14")   # 3.14
str(3)          # "3"
list("abc")     # ['a', 'b', 'c']
list((1, 2, 3)) # [1, 2, 3]
tuple([1, 2])   # (1, 2)
```

---

## 3. Control Flow

### 3.1 Conditionals (`if` / `elif` / `else`)

```python
x = 5

if x > 0:
    print("positive")
elif x < 0:
    print("negative")
else:
    print("zero")
```

**Comparison operators:** `>`, `<`, `==`, `!=`, `>=`, `<=`

**Logical operators:**

```python
if x > 0 and x < 10:   # both must be true
    print("single digit positive")

if x < 0 or x > 100:   # at least one must be true
    print("out of range")

if not x == 5:          # negation
    print("not five")
```

**One-liner (ternary):**

```python
label = "even" if x % 2 == 0 else "odd"
```

### 3.2 While Loop

Repeats while the condition is `True`.

```python
i = 0
while i < 5:
    print(i)
    i += 1   # same as i = i + 1
# prints 0 1 2 3 4
```

### 3.3 For Loop & `range`

Iterates over a sequence.

```python
# Iterate over a list
for item in [10, 20, 30]:
    print(item)

# range(stop)         → 0, 1, ..., stop-1
# range(start, stop)  → start, start+1, ..., stop-1
# range(start, stop, step)

for i in range(5):           # 0 1 2 3 4
    print(i)

for i in range(1, 101):      # 1 to 100
    print(i)

for i in range(0, 10, 2):    # 0 2 4 6 8
    print(i)

# Sum 1 to 100
total = 0
for i in range(101):
    total += i
print(total)  # 5050

# Nested loop — multiplication table
for i in range(1, 10):
    for j in range(1, 10):
        print(f"{i} * {j} = {i*j}")

# Iterate with index
for i, val in enumerate(["a", "b", "c"]):
    print(i, val)   # 0 a, 1 b, 2 c

# Iterate over two lists simultaneously
for a, b in zip([1, 2, 3], ["x", "y", "z"]):
    print(a, b)     # 1 x, 2 y, 3 z
```

### 3.4 `break` and `continue`

```python
# break — exit the loop immediately
i = 1
while True:
    print(i)
    i += 3
    if i >= 100:
        break

# continue — skip the rest of this iteration, go back to the top
for i in range(100):
    if i % 3 != 0:
        continue      # skip non-multiples of 3
    print(i)          # prints 0, 3, 6, ..., 99
```

---

## 4. Functions

### 4.1 Defining Functions

```python
# Basic structure
def function_name(parameters):
    # body
    return result     # optional

# Example
def add(a, b):
    return a + b

print(add(3, 5))  # 8
```

### 4.2 Return vs Print

```python
# return — passes a value back to the caller, ends the function
def square(x):
    return x * x

result = square(4)   # result = 16

# print — displays output; the function itself returns None
def show_square(x):
    print(x * x)

show_square(4)       # prints 16
val = show_square(4) # val is None
```

`return` inside a function behaves like `break` in a loop — it exits immediately.

```python
def check(x):
    if x % 2 == 0:
        return          # exit early (returns None)
    print(x, "is odd")

check(10)   # (nothing printed)
check(11)   # 11 is odd
```

### 4.3 Default Parameters

```python
def greet(name, msg="Hello"):
    print(f"{msg}, {name}!")

greet("Alice")          # Hello, Alice!
greet("Bob", "Hi")      # Hi, Bob!
```

### 4.4 Scope — Local vs Global

```python
a = 10           # global variable

def f():
    a = 20       # local variable — shadows global inside the function
    print(a)     # 20

f()
print(a)         # 10 — global is unchanged

# To modify the global variable inside a function:
def g():
    global a
    a = 20
    print(a)     # 20

g()
print(a)         # 20 — global is now changed
```

### 4.5 Recursion

A function that calls itself. Must have a **base case** to stop.

```python
# Fibonacci
def fibo(n):
    if n < 3:
        return 1
    return fibo(n - 1) + fibo(n - 2)

print(fibo(7))  # 13

# Factorial
def factorial(n):
    if n == 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))  # 120
```

---

## 5. Comprehensions

A concise way to build collections using a single line.

### 5.1 List Comprehension

```python
# Syntax: [expression for item in iterable if condition]

# Standard way
result = []
for i in range(100):
    if i % 2 == 0:
        result.append(2 * i)

# List comprehension
result = [2 * i for i in range(100) if i % 2 == 0]
# [0, 4, 8, 12, ..., 198]

# Without condition
squares = [x**2 for x in range(6)]
# [0, 1, 4, 9, 16, 25]

# Nested (flattening a 2D list)
matrix = [[1, 2], [3, 4], [5, 6]]
flat = [x for row in matrix for x in row]
# [1, 2, 3, 4, 5, 6]
```

### 5.2 Dictionary Comprehension

```python
# Syntax: {key_expr: value_expr for item in iterable if condition}

squares = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Invert a dictionary
original = {"a": 1, "b": 2, "c": 3}
inverted = {v: k for k, v in original.items()}
# {1: 'a', 2: 'b', 3: 'c'}
```

---

## 6. Object-Oriented Programming

### 6.1 Class Basics

```python
class Student:
    def __init__(self, name, score):   # constructor — called on object creation
        self.name = name               # instance variable
        self.score = score

    def show(self):
        print(f"{self.name}: {self.score}")

    def pass_check(self):
        return self.score >= 60

s1 = Student("Alice", 90)
s2 = Student("Bob", 45)

s1.show()              # Alice: 90
print(s2.pass_check()) # False
```

- `__init__` runs automatically when the object is created.
- `self` refers to the current instance and must be the first parameter of every method.

### 6.2 Inheritance & Overriding

```python
class Parent:
    def __init__(self, name, score):
        self.name = name
        self.score = score

    def show(self):
        print(self.name, self.score)


class Child(Parent):
    def __init__(self, name, score, number):
        Parent.__init__(self, name, score)   # call parent constructor
        self.number = number

    def show(self):                          # override parent's method
        print(self.number, self.name, self.score)

    def say_hello(self):
        print("hi")


c = Child("Alice", 90, 1)
c.show()       # 1 Alice 90
c.say_hello()  # hi
```

---

## 7. Modules

A module is a `.py` file that can be imported into other files.

### 7.1 Creating and Importing a Module

```python
# module.py
PI = 3.1415

def add(x, y):
    return x + y

def sub(x, y):
    return x - y

class Student:
    def __init__(self, name):
        self.name = name
    def show(self):
        print(self.name)

if __name__ == "__main__":
    print("running module.py directly")
    # code here only runs when module.py is executed directly,
    # NOT when it is imported by another file
```

```python
# main.py — three import styles

# Style 1: import the whole module
import module
print(module.PI)       # 3.1415
module.add(10, 5)      # 15
s = module.Student("Kim")
s.show()               # Kim

# Style 2: import specific names
from module import add, sub
print(add(10, 4))      # 14

# Style 3: import everything (use carefully — can cause name conflicts)
from module import *
print(PI)              # 3.1415
add(10, 3)             # 13
```

### 7.2 Module in a Folder (Package)

```
project/
├── main.py
└── moduleFolder/
    ├── __init__.py    ← makes the folder a package
    └── module.py
```

```python
# main.py — four import styles for packages

import moduleFolder.module
print(moduleFolder.module.PI)       # 3.1415

from moduleFolder.module import PI
print(PI)                           # 3.1415

from moduleFolder import module
print(module.PI)                    # 3.1415

from moduleFolder.module import *   # import all
```

Relative imports (inside the package):

```python
# moduleFolder2/module2.py
from ..moduleFolder import module   # .. = parent directory
```

---

## 8. Exception Handling

Prevents crashes by catching and handling errors at runtime.

```python
# Basic structure
try:
    statement1          # attempt this
except:
    statement2          # runs if any error occurs in try
finally:
    statement3          # always runs regardless of error

# Catching a specific error type
try:
    result = 4 / 0
except ZeroDivisionError as e:
    print(e)            # division by zero
finally:
    print("done")       # always printed

# Multiple except clauses
try:
    x = int("abc")
except ValueError as e:
    print("Value error:", e)
except TypeError as e:
    print("Type error:", e)

# Raising an exception manually
def set_age(age):
    if age < 0:
        raise ValueError("Age cannot be negative")
    return age
```

Common built-in exceptions:

| Exception           | Triggered by                            |
|---------------------|-----------------------------------------|
| `ZeroDivisionError` | Division by zero                        |
| `ValueError`        | Invalid value (e.g., `int("abc")`)      |
| `TypeError`         | Wrong type for an operation             |
| `IndexError`        | List index out of range                 |
| `KeyError`          | Dictionary key not found                |
| `FileNotFoundError` | Opening a file that doesn't exist       |
