### 1. Flexible Arguments (`*args` & `**kwargs`)

#### `*args` (Positional Arguments)

Allows a function to accept any number of positional arguments, storing them as a **tuple**.

* **Basic approach (Week 3):** `def add(a, b): return a + b` (Fails if you want to add 3 numbers).
* **Advanced approach:**
```python
def add_many(*args):
    # args behaves like a tuple: (1, 2, 3, ...)
    result = 0
    for num in args:
        result += num
    return result

print(add_many(1, 2, 3))       # Output: 6
print(add_many(10, 20, 30, 40)) # Output: 100
```
#### `**kwargs` (Keyword Arguments)

Allows a function to accept any number of keyword arguments (key-value pairs), storing them as a **dictionary**.

* **Example:**
```python
def introduce(**kwargs):
    # kwargs behaves like: {'name': 'Alice', 'age': 30}
    for key, value in kwargs.items():
        print(f"{key}: {value}")

introduce(name="Alice", age=30, city="Seoul")



```

---



### 2. Decorators (`@` Annotation)

 A **Decorator** allows you to modify the behavior of a function without changing its code directly.

* **Syntax:** uses the `@` symbol above a function definition.
* **Use Case:** Logging, measuring execution time, or checking login status.

**Example: Measuring Run Time**

```python
import time

# 1. Define the decorator (The Wrapper)
def timer_decorator(func):
    def wrapper():
        start_time = time.time()
        func()  # Run the original function
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")
    return wrapper

# 2. Apply the decorator
@timer_decorator
def heavy_calculation():
    # Simulating a Week 2 loop for calculation
    total = 0
    for i in range(1000000):
        total += i
    print("Calculation Done")

# Execution

heavy_calculation()

# Output:

# Calculation Done

# Execution time: 0.05 seconds

```

---



### 3. NumPy (Numerical Python)

In **Week 1** and **Week 4**, you learned about standard Lists (`[1, 2, 3]`) and basic operators (`+`, `*`).
While standard lists are flexible, they are slow for heavy math. **NumPy** is an external library used for high-performance scientific computing.

#### Why use NumPy over Lists?

1. **Speed:** NumPy arrays are stored in contiguous memory (unlike scattered Python lists).
2. **Element-wise Operations:** You cannot multiply a standard list by a number directly (e.g., `[1, 2] * 3` results in `[1, 2, 1, 2, 1, 2]`). NumPy handles this mathematically.

**Comparison Example:**

| Feature | Standard List (Week 4) | NumPy Array (Advanced) |
| --- | --- | --- |
| **Creation** | `a = [1, 2, 3]` | `import numpy as np`
`a = np.array([1, 2, 3])` |
| **Math (`a * 2`)** | `[1, 2, 3, 1, 2, 3]` (Repeats) | `[2, 4, 6]` (Calculates) |
| **Dimensions** | List inside List | N-Dimensional Array (Matrix) |

**Code Example:**

```python
import numpy as np

# Creating a 2D Matrix (Similar to Week 2 "2D Lists")
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# Mathematical Operations
print(matrix + 10) 
# Output (Adds 10 to EVERY element instantly):
# [[11, 12, 13],
#  [14, 15, 16]]

```

---

### 4. Lambda Functions

In **Week 3**, you defined functions using `def`. **Lambda** functions are small, anonymous functions defined in a single line. They are often used for short, one-time operations.

* **Syntax:** `lambda arguments : expression`

**Example:**

```python
# Week 3 Style
def add(x, y):
    return x + y
# Advanced Lambda Style
add_lambda = lambda x, y : x + y

print(add_lambda(3, 5)) # Output: 8

```

**Practical Use with `map()`:**
Applying a function to every item in a list (similar to a loop).

```python
nums = [1, 2, 3, 4]
squared = list(map(lambda x: x**2, nums))
print(squared) # Output: [1, 4, 9, 16]

```
---

### 5. Generators (`yield`)

In **Week 2**, you used `range(100)` or lists to iterate. If you create a list with 1 billion items, your computer might run out of memory. **Generators** produce items one by one only when asked, saving memory.

* **Keyword:** Uses `yield` instead of `return`.
* **Behavior:** The function pauses and saves its state after yielding.

**Example:**

```python
def my_generator():
    yield 1
    yield 2
    yield 3

gen = my_generator()

print(next(gen)) # Output: 1
print(next(gen)) # Output: 2
# Unlike 'return', the function didn't end; it paused.

```
Based on the style and depth of your existing `advanced.md` file, here is a new section on **Magic Methods (Dunder Methods)**. This includes the requested `__call__` and other essential special methods like `__init__` and `__str__`.

You can append this directly to the end of your file.

---

### 6. Magic Methods ("Dunder" Methods)

In Python, methods that start and end with double underscores (e.g., `__init__`) are called **Dunder Methods** (Double UNDERscore) or **Magic Methods**. They allow your custom objects to behave like built-in Python types (like lists or numbers).

#### Common Magic Methods

1. **`__init__` (Constructor):** Runs automatically when a new object is created.
2. **`__str__` (String Representation):** Runs when you pass an object to `print()`.
3. **`__call__` (Callable Object):** Allows an **instance** of a class to be called like a function.
4. **`__enter__` / `__exit__` (Context Manager):** Defines what happens when an object is used in a `with` statement, ensuring resources are properly managed.

**Code Example:**

```python
class SmartCounter:
    # 1. __init__: Setup initial state
    def __init__(self, start=0):
        self.count = start

    # 2. __call__: Makes the object 'callable' like a function
    def __call__(self, increment=1):
        self.count += increment
        print(f"Current count is now: {self.count}")

    # 3. __str__: distinct text when printing the object
    def __str__ (self):
        return f"A SmartCounter object with value: {self.count}"

# Usage
counter = SmartCounter(10)  # Runs __init__

print(counter)              
# Output (Runs __str__): A SmartCounter object with value: 10

# Calling the OBJECT variable as if it were a function
counter(5)                  
# Output (Runs __call__): Current count is now: 15

counter(2)

# Output (Runs __call__): Current count is now: 17



```



#### Why is `__call__` useful?

It allows you to maintain **state** (memory of previous values) inside what looks like a simple function call. This is often used in **Decorators** (Section 2) and machine learning frameworks (like PyTorch) to define layers.
---

#### Bonus: Operator Overloading (`__add__`)

Remember in **Section 3 (NumPy)** how we saw `matrix + 10`? You can teach your own classes how to use math symbols using dunder methods.

* `__add__` controls `+`
* `__sub__` controls `-`
* `__eq__` controls `==`

**Example:**

```python
class Wallet:
    def __init__(self, money):
        self.money = money

    # Teach the class how to use '+'
    def __add__(self, other):
        # 'other' is the second Wallet being added
        new_amount = self.money + other.money
        return Wallet(new_amount)
    
    def __str__(self):
        return f"${self.money}"

w1 = Wallet(50)
w2 = Wallet(100)

w3 = w1 + w2  # This triggers w1.__add__(w2)
print(w3)     # Output: 50

```
---

### 7. Context Managers (`with` statement)

A **Context Manager** is an object that defines a temporary context for a block of code. It ensures that resources are properly managed, such as automatically closing a file or releasing a lock.

* **Keyword:** The `with` statement is used to enter the context.
* **Use Case:** File handling, database connections, managing locks in multithreading.

**Traditional Way (Risky):**
If an error occurs while writing to the file, the `f.close()` line might never be reached, leaving the file open.

```python
f = open("my_file.txt", "w")
f.write("Hello")
# What if an error happens here?
f.close() # This might not get called
```

**Advanced Way (Safe and Clean):**
The `with` statement guarantees that the file will be closed automatically, even if errors occur inside the block.

```python
with open("my_file.txt", "w") as f:
    f.write("Hello")
# The file is automatically closed here

```

**How it Works (Behind the Scenes):**
Context managers are implemented using two magic methods:
* **`__enter__`**: Executed when entering the `with` block. It sets up the resource.
* **`__exit__`**: Executed when exiting the `with` block. It cleans up the resource (e.g., closes the file). This method always runs, regardless of errors.

You can create your own context managers by defining a class with these two methods.
---
### 8. Type Hinting

**Type Hinting** allows you to declare the expected data types of variables, function arguments, and return values. It does not enforce types (Python remains dynamically typed), but it provides valuable information for developers and tools.

* **Purpose:**
    * **Clarity:** Makes code easier to read and understand.
    * **Error Checking:** Static analysis tools (like Mypy) can catch type-related bugs before you run the code.
    * **IDE Support:** Improves auto-complete and code navigation.

**Syntax:**

* `variable: type`
* `def function(argument: type) -> return_type:`

**Example:**

```python
# No type hints (Hard to know what 'name' and 'age' should be)
def greet(name, age):
    return f"Hello, {name}. You are {age} years old."

# With type hints (Clear and explicit)
def greet_typed(name: str, age: int) -> str:
    return f"Hello, {name}. You are {age} years old."

# The hints make it obvious how to use the function
print(greet_typed("Alice", 30))

# An editor or type checker could warn you about this incorrect usage
# print(greet_typed(123, "thirty"))
```
**Common Types from the `typing` module:**
For more complex types, you can import them from the `typing` module.

* `List`: A list of a specific type (e.g., `List[int]`)
* `Dict`: A dictionary with specific key and value types (e.g., `Dict[str, float]`)
* `Tuple`: A tuple with specific types
* `Optional`: For a value that could be `None` (e.g., `Optional[str]`)

```python
from typing import List, Optional

def process_scores(scores: List[int]) -> None:
    for score in scores:
        print(f"Processing score: {score}")

def find_user(user_id: str) -> Optional[str]:
    if user_id == "admin":
        return "Administrator"
    return None # It's valid to return None
```