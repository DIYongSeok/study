# Python Basics Note (Week 1 - Week 6)

## <Week 1> Data Types & Basic Operations

### 1. Numeric Data
* **Examples:** `1`, `2` (Integer), `3.100` (Float).
* **Operators:**
    * `+`, `-`, `*`, `/`
    * `**` (Exponent/Power), `%` (Modulo/Remainder)
* **Example:** `3 + 3` $\rightarrow$ `6`

### 2. String Data
* **Examples:** `"3"`, `'3'`
* **Concatenation:** `"3" + "3"` $\rightarrow$ `"33"` (Strings are joined, not added).
* **Type Conversion:**
    * `int("3")` $\rightarrow$ `3`
    * `str(3)` $\rightarrow$ `"3"`

### 3. Input & Processing
* **Input:**
    ```python
    i = input()       # Receive input from user
    ans = i.split()   # Split string into a list based on spaces
    
    # Example: User inputs "10 20"
    ans2 = int(ans[0]) + int(ans[1]) # 10 + 20
    print(ans2) # Output: 30
    ```

### 4. List Data
* **Definition:** A collection of items. Example: `a = [1, 2, 3, 4]`
* **Slicing (Accessing parts of a list):**
    * `a[1:3]`: Returns items from index 1 up to (but not including) 3.
    * `a[:]`: Returns all items.
    * `a[:3]`: Returns items from the beginning up to index 3.
    * `a[1:]`: Returns items from index 1 to the end.
* **Methods:**
    * `len(a)`: Returns the number of items.
    * `a.append(10)`: Adds `10` to the end.
    * `a.pop()`: Removes the last item.
    * `a.sort()`: Sorts values in ascending order.
    * `a.reverse()`: Reverses the order of items.
* **String Split Example:**
    * `"hello world".split()` $\rightarrow$ `['hello', 'world']`

### 5. Tuple & Dictionary

#### Tuple
* **Definition:** Similar to a list, but **immutable** (cannot be changed). Defined with `()`.
* **When to use:** For data that should not be modified, like coordinates or fixed settings.
* **Example:**
    ```python
    my_tuple = (1, 2, 3)
    # my_tuple[0] = 10  # This would cause an error!
    ```

#### Dictionary
* **Definition:** A collection of **key-value pairs**. Defined with `{}`.
* **When to use:** For storing labeled data, like user profiles or settings.
* **Example:**
    ```python
    student = {
        "name": "Alice",
        "age": 25,
        "courses": ["Math", "Science"]
    }

    # Accessing data
    print(student["name"])  # Output: Alice

    # Adding new data
    student["grade"] = "A"
    ```

---

## <Week 2> Conditional Statements & Loops

### 1. Conditional Statements (`if`)
* **Structure:**
    ```python
    if condition1:
        statement1    # Executed if condition1 is True
    elif condition2:
        statement2    # Executed if condition1 is False and condition2 is True
    else:
        statement3    # Executed if all above are False
    ```
* **Comparison Operators:** `>`, `<`, `==` (equal), `!=` (not equal), `<=`, `>=`
* **Logical Operators:**
    * `and`: Both conditions must be true.
    * `or`: At least one condition must be true.

### 2. Loops (`while` / `for`)
* **While Loop:** Runs as long as the condition is true.
    ```python
    i = 0
    while i < 100:
        print(i)
        i += 1  # Increment (same as i = i + 1)
    ```
* **For Loop:** Iterates over a sequence (list or range).
    ```python
    _list = [3, 4, 10, 15]
    ans = 0
    for i in _list:
        ans += i
    print(ans) # Sum of list
    ```

### 3. Loop Control
* **`break`**: Immediately exits the loop.
* **`continue`**: Skips the rest of the current iteration and jumps back to the start of the loop.
    * *Example:* Printing odd numbers below 10.
        ```python
        for i in range(10):
            if i % 2 == 0:  # If the number is even
                continue    # Skip to the next number
            print(i)
        ```

### 4. 2D Lists
* **Structure:** List inside a list.
* **Example:**
    ```python
    a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # a[0][0] is 1
    # a[1][2] is 6
    ```

---

## <Week 3 & 4> Functions & Advanced Loops

### 1. Range Function
* `range(100)` $\rightarrow$ Generates `0` to `99`.
* `range(10, 100)` $\rightarrow$ Generates `10` to `99`.

### 2. Functions (`def`)
* **Structure:**
    ```python
    def function_name(input_value):
        # commands
        return output_value
    ```
* **Return vs Print:** `return` passes a value back and ends the function (similar to `break` in loops).

### 3. Nested Loops
* **Example: Multiplication Table (Gu-Gu-Dan)**
    ```python
    for i in range(1, 10):
        for j in range(1, 10):
            print(f"{i} * {j} = {i*j}")
    ```

### 4. Recursion
* A function that calls itself. Must have a base case (stopping condition).
* **Example: Fibonacci**
    ```python
    def fibo(n):
        if n < 3:
            return 1
        else:
            return fibo(n-1) + fibo(n-2)
    ```
* **Example: Factorial**
    ```python
    def factorial(n):
        if n == 1:
            return 1
        else:
            return n * factorial(n-1)
    ```

---

## <Week 5> Scope & Comprehensions

### 1. Variables: Local vs Global
* **Local Variable:** Exists only inside the function.
* **Global Variable:** Exists everywhere in the script.
* **Keyword `global`:** Used to modify a global variable inside a function.
    ```python
    a = 10
    def f():
        global a
        a = 20  # Modifies the global 'a'
    ```

### 2. List & Dictionary Comprehension

#### List Comprehension
* A concise way to create lists.
* **Syntax:** `[expression for item in list if condition]`
* **Example:**
    ```python
    # Normal way
    result = []
    for i in range(10):
        if i % 2 == 0:
            result.append(i**2)
            
    # List Comprehension way
    result = [i**2 for i in range(10) if i % 2 == 0] # Output: [0, 4, 16, 36, 64]
    ```

#### Dictionary Comprehension
* A concise way to create dictionaries.
* **Syntax:** `{key_expression: value_expression for item in list if condition}`
* **Example:**
    ```python
    # Create a dictionary with numbers and their squares
    squares = {i: i**2 for i in range(5)}
    # Output: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
    ```

---

## <Week 6> Classes, Modules, & Exceptions

### 1. Classes (Object Oriented Programming)
* **Definition:**
    ```python
    class Student:
        def __init__(self, name):
            self.name = name  # Instance variable
            
        def show(self):
            print(self.name)
    ```
* **`__init__`**: Constructor method, runs automatically when an object is created.
* **`self`**: Represents the instance of the object itself.

### 2. Inheritance
* Classes can inherit properties from other classes.
* **Overriding:** Re-writing a method in the child class.
    ```python
    class Parent:
        def show(self): print("Parent")
        
    class Child(Parent):
        def show(self): print("Child") # Overrides Parent's show
    ```

### 3. Modules
* **Files:** `module.py` (contains functions/variables), `main.py` (imports module).
* **Import Styles:**
    1.  `import module` $\rightarrow$ `module.add(1, 2)`
    2.  `from module import add` $\rightarrow$ `add(1, 2)`
    3.  `from module import *` $\rightarrow$ Import all.
* **Execution Control:**
    ```python
    if __name__ == "__main__":
        print("This runs only if file is executed directly")
    ```

### 4. Exception Handling
* Prevents program crashes on errors.
    ```python
    try:
        4 / 0
    except ZeroDivisionError as e:
        print(f"Error occurred: {e}")
    finally:
        print("This always runs")
    ```