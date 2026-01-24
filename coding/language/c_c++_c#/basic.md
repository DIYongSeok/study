### **1. C Language**

An imperative, procedural language, C is the foundation for many other programming languages. It's known for its performance and low-level memory access.

#### **1.1 Pointers**

*   **Concept:** Pointers are variables that store the memory address of another variable. They are fundamental to C for tasks like dynamic memory allocation and efficient array handling. An array's name can be treated as a constant pointer to its first element.

*   **Example 1: Basic Declaration**

    ```c
    int a = 5;
    int *b = &a;  // 'b' stores the memory address of 'a'
    int **c = &b; // 'c' stores the memory address of the pointer 'b'

    printf("Value of a: %d\n", a);
    printf("Value at address stored in b: %d\n", *b);
    printf("Value at address stored in c's pointed-to address: %d\n", **c);
    ```

*   **Example 2: Call by Value vs. Call by Reference**

    *   **(1) Call by Value:** Passes a copy of the variable's value to the function. Any modifications inside the function do not affect the original variable.

        ```c
        void add(int x) {
            x += 10;
        }

        // Usage:
        int x = 5;
        add(x); // 'x' remains 5
        ```

    *   **(2) Call by Reference (via Pointers):** Passes the memory address of the variable. The function can then dereference the pointer to modify the original variable's value.

        ```c
        void add_by_ref(int *x) {
            *x += 10;
        }

        // Usage:
        int x = 5;
        add_by_ref(&x); // 'x' becomes 15
        ```

#### **1.2 Memory Layout in a Process**

When a C program is executed, the operating system allocates memory for it, typically divided into four main segments:

1.  **Code (or Text) Segment:** Stores the compiled machine code of the program. This area is often read-only.
2.  **Data Segment:** Stores global variables and static variables. This segment can be further divided into initialized (e.g., `int global_var = 10;`) and uninitialized (BSS) data.
3.  **Heap Segment:** Used for dynamic memory allocation. Memory is requested by the programmer using `malloc`, `calloc`, `realloc` and must be explicitly freed using `free`.
4.  **Stack Segment:** Used for local variables, function parameters, and return addresses. Memory is automatically allocated and deallocated as functions are called and return (LIFO - Last-In, First-Out).

#### **1.3 Variable Storage Classes**

*   **Static Variables:** A static variable's lifetime persists for the entire duration of the program, even if declared inside a function. It retains its value between function calls.

    ```c
    #include <stdio.h>
    void process() {
        static int a = 5; // Initialized only once
        a += 1;
        printf("%d ", a);
    }

    // Calling process() three times will print: 6 7 8
    ```

*   **Register Variables:** A hint to the compiler to store the variable in a CPU register instead of RAM for faster access. The compiler may ignore this suggestion. It's less commonly used in modern C due to advanced compiler optimizations.

    ```c
    register int counter;
    ```

#### **1.4 Dynamic Memory Allocation**

*   **Concept:** Allows allocating memory at runtime, which is essential when the required memory size is not known at compile time.

*   **Example:**

    ```c
    #include <stdio.h>
    #include <stdlib.h>

    int n = 5;
    // int a[n]; // This is a Variable Length Array (VLA), a C99 feature, but not standard in C89 or C++.

    // The standard way for dynamic allocation:
    int *a = (int*)malloc(sizeof(int) * n); // Allocate memory for 5 integers
    if (a == NULL) {
        // Always check if malloc was successful
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    for(int i = 0; i < n; i++) {
        a[i] = i + 1;
    }

    free(a); // Always release the memory to prevent memory leaks.
    a = NULL; // Good practice to set pointer to NULL after freeing.
    ```

#### **1.5 Structs (Structures)**

*   **Concept:** A user-defined data type that groups together variables of different data types. It allows you to create complex data structures.

*   **Example:**

    ```c
    #include <stdio.h> 
    #include <stdlib.h> 

    struct Student {
        int id;
        int score;
    }; // Don't forget the semicolon

    // Declaration and Initialization
    struct Student s1 = {10207, 95};

    // Accessing members
    printf("Student ID: %d\n", s1.id);

    // Using pointers with structs
    struct Student *ptr = (struct Student*)malloc(sizeof(struct Student));
    if (ptr != NULL) {
        ptr->id = 10208;    // Use '->' (arrow operator) to access members via a pointer
        ptr->score = 88;
        printf("Pointer to Student ID: %d\n", ptr->id);
        free(ptr);
    }
    ```

#### **1.6 Preprocessor & Modularization**

*   **#include:** Inserts the content of another file.
    *   `#include <stdio.h>`: Searches in standard system directories first. Used for system libraries.
    *   `#include "myheader.h"`: Searches in the current project directory first. Used for your own project's header files.

*   **#define:** A preprocessor directive used to define macros.
    ```c
    #define PI 3.14159
    #define SQUARE(x) ((x) * (x)) // Parentheses are crucial to avoid operator precedence issues
    ```

*   **Header Guards (#ifndef):** A standard C practice to prevent the same header file from being included multiple times in a single compilation unit, which would cause "duplicate definition" errors.
    ```c
    // myheader.h
    #ifndef MYHEADER_H
    #define MYHEADER_H

    // All your declarations go here
    int add(int x, int y);

    #endif // MYHEADER_H
    ```

---

### **2. C++ Language**

C++ is a high-level, general-purpose language created as an extension of C. It introduces object-oriented features like classes, inheritance, and polymorphism.

#### **2.1 Basic I/O & Dynamic Allocation**

*   **I/O:** Uses `iostream` header for type-safe I/O with `std::cin` (input) and `std::cout` (output).

*   **Dynamic Allocation:** Uses `new` to allocate memory and `delete` to deallocate it. They are type-safe and automatically call object constructors and destructors.

    ```cpp
    #include <iostream>
    #include <string>

    int main() {
        std::string input;
        std::cout << "Enter text: ";
        std::cin >> input;
        std::cout << "You entered: " << input << std::endl;

        int* arr = new int[100]; // Dynamic allocation for an array
        delete[] arr;            // Deallocation for an array
        arr = nullptr;           // Good practice

        int* single_int = new int;
        delete single_int;       // Deallocation for a single object
        single_int = nullptr;

        return 0;
    }
    ```

#### **2.2 Classes**

*   **Concept:** The core of C++ OOP. A class is a blueprint for creating objects. It bundles data (attributes) and methods (functions) that operate on that data. Access is controlled by `public`, `private`, and `protected` specifiers.

*   **Example:**

    ```cpp
    #include <iostream>
    #include <string>

    class Student {
    private:
        std::string name;
        int score;

    public:
        // Constructor with member initializer list (more efficient)
        Student(std::string n, int s) : name(n), score(s) {
            // 'this' pointer is implicitly used here
        }

        void show() {
            std::cout << this->name << " has a score of " << this->score << std::endl;
        }
    };
    ```

#### **2.3 Inheritance & The Rule of Three/Five**

*   **Inheritance:** Allows a new class (derived/child) to inherit properties and methods from an existing class (base/parent). Use the colon `:` for inheritance.

*   **The Rule of Three/Five & Deep Copy:** If a class manages a raw resource (like memory via a raw pointer), you must provide:
    1.  **Copy Constructor:** To create a proper "deep copy" with its own resource.
    2.  **Copy Assignment Operator:** To handle assignment (`obj1 = obj2;`) correctly.
    3.  **Destructor:** To release the resource.
    (C++11 adds Move Constructor and Move Assignment Operator, making it the "Rule of Five").

    ```cpp
    // Example of a class needing a deep copy
    class MyArray {
    private:
        int* data;
        int size;
    public:
        // Constructor
        MyArray(int s) : size(s), data(new int[s]) {}

        // Destructor
        ~MyArray() {
            delete[] data;
        }

        // 1. Copy Constructor (Deep Copy)
        MyArray(const MyArray& other) : size(other.size), data(new int[other.size]) {
            for (int i = 0; i < size; ++i) {
                data[i] = other.data[i];
            }
        }

        // 2. Copy Assignment Operator
        MyArray& operator=(const MyArray& other) {
            if (this == &other) return *this; // Self-assignment check
            delete[] data; // Free old memory
            size = other.size;
            data = new int[size];
            for (int i = 0; i < size; ++i) {
                data[i] = other.data[i];
            }
            return *this;
        }
    };
    ```

#### **2.4 Polymorphism (Virtual Functions)**

*   **Concept:** Allows objects of different classes to be treated as objects of a common base class.
*   **Static Binding (Compile-time):** The default behavior. When a base class pointer holds a child object, calling a method results in the *base class's* version being executed.
*   **Dynamic Binding (Runtime):** Achieved using the `virtual` keyword. The program decides at runtime which version of the function to call based on the object's actual type.

    ```cpp
    #include <iostream>
    class A {
    public:
        // Virtual function enables dynamic binding
        virtual void show() { std::cout << "A class" << std::endl; }
        virtual ~A() {} // Virtual destructor is crucial!
    };

    class B : public A {
    public:
        // 'override' is a good practice to ensure you're overriding a virtual function
        void show() override { std::cout << "B class" << std::endl; }
    };

    int main() {
        A* p = new B(); // A pointer to a B object
        p->show();      // Prints "B class" because show() is virtual
        delete p;       // Calls B's destructor then A's destructor
    }
    ```
    *   **Virtual Destructor:** Always make the base class destructor `virtual` if you intend to `delete` a derived object through a base class pointer. This ensures the correct chain of destructors is called, preventing resource leaks.

#### **2.5 Templates (Generics)**

*   **Concept:** Allow you to write generic functions and classes that can work with any data type, promoting code reusability. The compiler generates the specific version of the code needed at compile time.

*   **Example:**

    ```cpp
    template <typename T>
    class Data {
    private:
        T value;
    public:
        Data(T val) : value(val) {}
        T getData() const { return value; } // const-correctness is good practice
    };

    // Usage:
    Data<int> data1(10);
    Data<std::string> data2("hello");
    ```

#### **2.6 Smart Pointers (RAII)**

*   **Concept:** Wrapper classes for raw pointers that automate memory management based on the RAII (Resource Acquisition Is Initialization) principle. They help prevent memory leaks by ensuring memory is deallocated when the smart pointer goes out of scope. Found in the `<memory>` header.

*   **Types:**
    *   `std::unique_ptr`: Provides exclusive ownership. The memory is deallocated when the `unique_ptr` is destroyed. It cannot be copied, only moved.
    *   `std::shared_ptr`: Manages a resource shared among multiple owners. It keeps a reference count, and the memory is deallocated only when the last `shared_ptr` pointing to it is destroyed.
    *   `std::weak_ptr`: A non-owning pointer that holds a weak reference to an object managed by a `shared_ptr`. It's used to break circular dependency issues.

*   **Example (`unique_ptr`):**

    ```cpp
    #include <memory>
    #include <iostream>

    void process_ptr(std::unique_ptr<int> ptr) {
        std::cout << "Processing pointer with value: " << *ptr << std::endl;
    }

    // Creating smart pointers is best done with std::make_unique / std::make_shared
    std::unique_ptr<int> p1 = std::make_unique<int>(10);
    std::unique_ptr<int> p2;

    // p2 = p1; // Compile Error: unique_ptr cannot be copied
    p2 = std::move(p1); // Ownership is transferred from p1 to p2. p1 is now nullptr.

    process_ptr(std::move(p2)); // Ownership is transferred to the function
    // p2 is now nullptr
    ```

#### **2.7 Exception Handling**

*   **Structure:** A mechanism for handling runtime errors in a structured way.
    *   `try`: Encloses a block of code that might throw an exception.
    *   `throw`: Used to signal that an error has occurred.
    *   `catch`: Catches and handles the thrown exception.

    ```cpp
    #include <iostream>

    double divide(int a, int b) {
        if (b == 0) {
            throw "Error: Cannot divide by zero!";
        }
        return static_cast<double>(a) / b;
    }

    try {
        double result = divide(10, 0);
        std::cout << "Result: " << result << std::endl;
    }
    catch (const char* error_message) {
        std::cerr << error_message << std::endl; // Prints the error message
    }
    ```

#### **2.8 Operator Overloading**

*   **Concept:** Allows you to redefine how standard operators (like `+`, `-`, `*`, `<<`) behave with user-defined types (classes or structs).

    ```cpp
    struct Position {
        int x, y;

        Position operator+(const Position& other) {
            Position new_pos;
            new_pos.x = this->x + other.x;
            new_pos.y = this->y + other.y;
            return new_pos;
        }
    };
    ```

---

### **3. C# Language**

C# is a modern, object-oriented, and type-safe programming language developed by Microsoft. It runs on the .NET Framework and is commonly used for building Windows applications, web services, and games (with Unity).

#### **3.1 Basic Syntax**

*   **Hello World:**

    ```csharp
    // C# 9 and later provides top-level statements for simpler programs:
    // Console.WriteLine("Hello, World!");

    // Traditional structure:
    namespace ConsoleApp
    {
        class Program
        {
            static void Main(string[] args)
            {
                Console.WriteLine("Hello, World!");
            }
        }
    }
    ```

*   **Arrays:**
    *   Single-dimensional: `int[] arr = new int[4];`
    *   Multi-dimensional (Rectangular): `int[,] arr = new int[2, 4];`
    *   Jagged (Array of arrays): `int[][] arr = new int[2][]; arr[0] = new int[3]; arr[1] = new int[5];`

*   **Foreach Loop:** A simple way to iterate over collections.

    ```csharp
    int[] numbers = { 1, 2, 3, 4, 5 };
    foreach(int i in numbers)
    {
        Console.WriteLine(i);
    }
    ```

#### **3.2 Classes vs. Structs**

This is a key distinction in C#:

*   **Struct:**
    *   **Value Type:** Stored on the stack (or inline in containing objects).
    *   When you pass a struct to a method, a **copy** of the struct is passed.
    *   Cannot have a parameterless constructor (one is provided by default).
    *   Cannot inherit from another class or struct (but can implement interfaces).
    *   Ideal for small, lightweight data containers.

*   **Class:**
    *   **Reference Type:** An object is created on the heap, and a reference (pointer) to it is stored on the stack.
    *   When you pass a class object to a method, a **copy of the reference** is passed. Both the original reference and the copy point to the *same* object on the heap.
    *   Supports inheritance

#### **3.3 Inheritance & Polymorphism**

*   **Keywords:**
    *   `virtual`: In the base class, allows a method to be overridden by a derived class.
    *   `override`: In the derived class, provides a new implementation for a `virtual` method from the base class.
    *   `base`: Used to access members of the base class from within a derived class.
    *   `sealed`: When applied to a class, prevents other classes from inheriting from it. When applied to a method, prevents it from being overridden further.

*   **Example:**

    ```csharp
    public class Monster
    {
        public virtual void Attack() { Console.WriteLine("A generic monster attacks!"); }
    }

    public class Orc : Monster
    {
        public override void Attack()
        {
            base.Attack(); // Optionally calls the parent's Attack method
            Console.WriteLine("The Orc swings its club!");
        }
    }
    ```

#### **3.4 Collections (Generics)**

The `System.Collections.Generic` namespace provides strongly-typed collections.

*   **List<T> (Dynamic Array):** A versatile, dynamically-sized list.

    ```csharp
    List<string> names = new List<string>();
    names.Add("Alice");
    names.Add("Bob");
    names.Remove("Alice"); // Removes the first occurrence of "Alice"
    ```

*   **Dictionary<TKey, TValue> (Map):** A collection of key-value pairs.

    ```csharp
    Dictionary<string, int> studentScores = new Dictionary<string, int>();
    studentScores["Alice"] = 95;
    studentScores["Bob"] = 87;

    foreach(KeyValuePair<string, int> pair in studentScores)
    {
        Console.WriteLine($"{pair.Key}: {pair.Value}"); // String interpolation
    }
    ```

#### **3.5 Advanced & Modern Features**

*   **Properties:** A more expressive way to provide access to class fields than traditional get/set methods.

    ```csharp
    public class Player
    {
        // Auto-implemented property
        public int Health { get; set; } = 100;

        private string _name;
        // Full property with backing field and validation
        public string Name
        {
            get { return _name; }
            set
            {
                if (!string.IsNullOrEmpty(value))
                {
                    _name = value;
                }
            }
        }
    }
    ```

*   **Delegates, Events, and Lambdas:** These features are the foundation for event-driven programming and LINQ.
    *   **Delegate:** A type that represents a reference to a method with a specific signature. It's like a function pointer in C++.
    *   **Lambda Expression:** An anonymous function that provides a concise way to create instances of delegates or expression trees. `(parameters) => expression-or-statement-block`
    *   **Event:** A specialized delegate that provides a safe way for a class to expose notifications to clients. Clients can add or remove event handlers (`+=` and `-=`).

    ```csharp
    public class Button
    {
        // 1. Define a delegate for the event
        public delegate void ClickEventHandler(object sender, EventArgs e);
        // 2. Define the event based on the delegate
        public event ClickEventHandler Clicked;

        // 3. Method to raise the event
        public void SimulateClick()
        {
            Clicked?.Invoke(this, EventArgs.Empty); // Safely invoke the event
        }
    }

    // Usage:
    var myButton = new Button();
    // 4. Subscribe to the event using a lambda expression
    myButton.Clicked += (sender, args) => Console.WriteLine("Button was clicked!");
    myButton.SimulateClick();
    ```
