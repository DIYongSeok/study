# C++ Basic

## Table of Contents

1. [Program Structure](#1-program-structure)
   - 1.1 [I/O](#11-io)
   - 1.2 [Namespace](#12-namespace)
   - 1.3 [Scope Resolution Operator`::`](#13-scope-resolution-operator-)
2. [Variables & Memory](#2-variables--memory)
   - 2.1 [Data Types & Literals](#21-data-types--literals)
   - 2.2 [Dynamic Memory Allocation](#22-dynamic-memory-allocation)
   - 2.3 [Enum](#23-enum)
3. [Control Flow](#3-control-flow)
   - 3.1 [Conditionals](#31-conditionals)
   - 3.2 [Loops](#32-loops)
   - 3.3 [Exception Handling](#33-exception-handling)
4. [Functions](#4-functions)
   - 4.1 [Basic Functions](#41-basic-functions)
   - 4.2 [Function Overloading](#42-function-overloading)
   - 4.3 [Default Parameters](#43-default-parameters)
5. [Object-Oriented Programming](#5-object-oriented-programming)
   - 5.1 [Class Basics](#51-class-basics)
   - 5.2 [Access Modifiers](#52-access-modifiers)
   - 5.3 [Constructors & Destructor](#53-constructors--destructor)
   - 5.4 [Initializer List](#54-initializer-list)
   - 5.5 [Static Members](#55-static-members)
   - 5.6 [Const Members & Methods](#56-const-members--methods)
   - 5.7 [Friend](#57-friend)
6. [Inheritance & Polymorphism](#6-inheritance--polymorphism)
   - 6.1 [Inheritance](#61-inheritance)
   - 6.2 [Virtual Functions](#62-virtual-functions)
   - 6.3 [Pure Virtual Functions & Abstract Class](#63-pure-virtual-functions--abstract-class)
   - 6.4 [Operator Overloading](#64-operator-overloading)
7. [Templates](#7-templates)
   - 7.1 [Function Template](#71-function-template)
   - 7.2 [Explicit Specialization](#72-explicit-specialization)
   - 7.3 [Class Template](#73-class-template)
8. [Smart Pointers](#8-smart-pointers)
   - 8.1 [unique_ptr](#81-unique_ptr)
   - 8.2 [shared_ptr & weak_ptr](#82-shared_ptr--weak_ptr)
9. [STL Basics](#9-stl-basics)
   - 9.1 [Stack](#91-stack)
   - 9.2 [Queue](#92-queue)
   - 9.3 [Vector](#93-vector)

---

## 1. Program Structure

### 1.1 I/O

```cpp
#include <iostream>
#include <string>
using namespace std;

int main() {
    string input;
    cin >> input;              // reads one word (stops at whitespace)
    getline(cin, input);       // reads a full line including spaces
    cout << input << endl;
    return 0;
}
```

### 1.2 Namespace

Prevents name collisions when multiple libraries define the same name.

```cpp
namespace A {
    void function() { cout << "function A" << endl; }
}
namespace B {
    void function() { cout << "function B" << endl; }
}

A::function();  // function A
B::function();  // function B

using namespace A;  // allows omitting A::
```

### 1.3 Scope Resolution Operator `::`

- Access members of a namespace or class
- Access static members
- Access nested classes

```cpp
vector<int>::iterator iter;   // type inside a class
Person::count;                // static member
```

---

## 2. Variables & Memory

### 2.1 Data Types & Literals

```cpp
int    a = 10;
double b = 3.14;
char   c = 'A';
bool   d = true;
string s = "hello";

// type aliases
typedef long long ll;
#define ll long long
```

### 2.2 Dynamic Memory Allocation

```cpp
// Single value
int* p = new int(5);
delete p;

// Array
int n = 5;
int* arr = new int[n];
delete[] arr;        // must use delete[] for arrays

// With struct/class
Student* s = new Student("Alice", 90);
delete s;
```

> Always pair `new` with `delete` and `new[]` with `delete[]` to prevent memory leaks.

### 2.3 Enum

Maps names to integer values (starting from 0 by default).

```cpp
enum Color { Black, Green, Red, Blue };

Color c = Blue;
cout << c;  // 3

// Custom values
enum Status { OK = 200, NOT_FOUND = 404, ERROR = 500 };
```

---

## 3. Control Flow

### 3.1 Conditionals

```cpp
if (x > 0)      cout << "positive";
else if (x < 0) cout << "negative";
else            cout << "zero";

// Switch
switch (x) {
    case 1:  cout << "one"; break;
    case 2:  cout << "two"; break;
    default: cout << "other";
}
```

### 3.2 Loops

```cpp
for (int i = 0; i < 5; i++) cout << i;

while (x > 0) x--;

do { x--; } while (x > 0);

// Range-based for (C++11)
vector<int> v = {1, 2, 3};
for (int x : v)   cout << x;        // copy
for (int& x : v)  x *= 2;           // modify in-place
for (const int& x : v) cout << x;   // read-only
```

### 3.3 Exception Handling

```cpp
try {
    if (b == 0) throw "cannot divide by 0";
    cout << a / b;
}
catch (const char* msg) {
    cout << msg;
}
catch (...) {
    cout << "unknown error";  // catch-all
}
```

---

## 4. Functions

### 4.1 Basic Functions

```cpp
int add(int a, int b) {
    return a + b;
}

// Pass by value — caller's variable is unchanged
void increment(int x) { x++; }

// Pass by pointer — modifies original
void increment(int* x) { (*x)++; }

// Pass by reference — cleaner syntax, modifies original
void increment(int& x) { x++; }
```

### 4.2 Function Overloading

Same name, different parameter types or counts.

```cpp
void print(int x)    { cout << x; }
void print(double x) { cout << x; }
void print(string x) { cout << x; }
```

### 4.3 Default Parameters

```cpp
void greet(string name, string msg = "Hello") {
    cout << msg << ", " << name;
}

greet("Alice");           // Hello, Alice
greet("Bob", "Hi");       // Hi, Bob
```

---

## 5. Object-Oriented Programming

### 5.1 Class Basics

```cpp
class Student {
private:
    string name;
    int score;
public:
    Student(string n, int s) : name(n), score(s) {}

    void show() {
        cout << name << " " << score << endl;
    }
};

Student a("Alice", 90);
a.show();
```

- All instances share the same function memory; `this->` is automatically applied inside methods.
- Use `this->name = name` to distinguish member variable from a same-named parameter.

### 5.2 Access Modifiers

| Modifier    | Same Class | Subclass | Elsewhere |
|-------------|:----------:|:--------:|:---------:|
| `public`    | O          | O        | O         |
| `protected` | O          | O        | X         |
| `private`   | O          | X        | X         |

Members without an explicit modifier default to `private`.

### 5.3 Constructors & Destructor

```cpp
class Student {
    string name;
    int score;
public:
    // Default constructor
    Student() : name("unknown"), score(0) {}

    // Parameterized constructor
    Student(string name, int score) : name(name), score(score) {}

    // Copy constructor (deep copy)
    Student(const Student& other) : name(other.name), score(other.score) {}

    // Destructor — called automatically when object goes out of scope
    ~Student() { cout << "destroyed" << endl; }
};

Student a("Alice", 90);
Student b(a);                        // copy constructor
Student* c = new Student("Bob", 80); // heap allocation
delete c;                            // destructor called here
```

- Multiple constructors are allowed; only one destructor (no parameters, no overloading).
- If no constructor is defined, a default constructor with zero-initialized members is used.

### 5.4 Initializer List

Preferred over assignment inside the constructor body. Required for `const` members, reference members, and parent class constructors.

```cpp
// Problem — const cannot be assigned, parent called twice
Knight() {
    Player(1);   // warning: parent constructor called twice
    _hp = 10;    // error: cannot modify const member
}

// Solution — initializer list
Knight() : Player(1), _hp(10), _inven() {}
```

### 5.5 Static Members

Shared across all instances; belongs to the class, not any specific object.

```cpp
class Person {
public:
    static int count;
    Person() { count++; }
};

int Person::count = 0;  // must be defined outside the class

Person a, b, c;
cout << Person::count;  // 3  (or a.count — both work)
```

### 5.6 Const Members & Methods

```cpp
class Circle {
    const double PI = 3.14159;  // const member — set once at construction
    double radius;
public:
    Circle(double r) : radius(r) {}

    double area() const {       // const method — cannot modify any member
        return PI * radius * radius;
    }
};
```

### 5.7 Friend

Grants another class or function access to `private` members.

```cpp
class Box {
    friend class Inspector;  // Inspector can access Box's private members
    int secret = 42;
};

class Inspector {
public:
    void reveal(Box b) { cout << b.secret; }  // allowed
};
```

---

## 6. Inheritance & Polymorphism

### 6.1 Inheritance

```cpp
class Animal {
public:
    string name;
    Animal(string name) : name(name) {}
    void breathe() { cout << name << " breathes" << endl; }
};

class Dog : public Animal {
public:
    Dog(string name) : Animal(name) {}  // call parent constructor
    void bark() { cout << name << " barks" << endl; }
};

Dog d("Rex");
d.breathe();  // inherited from Animal
d.bark();
```

- Parent constructor runs first; destructors run in reverse (child → parent).
- `public` inheritance keeps parent's access levels intact.
- Multiple inheritance is possible but use with caution.

### 6.2 Virtual Functions

Without `virtual`, the function called depends on the **pointer type** (static binding):

```cpp
class A { public: void show() { cout << "A"; } };
class B : public A { public: void show() { cout << "B"; } };

A* p = new B();
p->show();  // prints "A" — wrong!
```

With `virtual` (dynamic binding — resolved at runtime):

```cpp
class A {
public:
    virtual void show() { cout << "A"; }
    virtual ~A() {}  // always make destructor virtual in polymorphic classes
};

class B : public A {
public:
    void show() override { cout << "B"; }
};

A* p = new B();
p->show();  // prints "B" — correct
delete p;   // B's destructor called first, then A's
```

### 6.3 Pure Virtual Functions & Abstract Class

Forces subclasses to provide an implementation. A class with any pure virtual function becomes abstract and cannot be instantiated.

```cpp
class Shape {
public:
    virtual void draw() = 0;    // pure virtual
    virtual double area() = 0;
    virtual ~Shape() {}
};

class Circle : public Shape {
    double r;
public:
    Circle(double r) : r(r) {}
    void draw() override { cout << "drawing circle"; }
    double area() override { return 3.14 * r * r; }
};

// Shape s;  // error — cannot instantiate abstract class
Shape* s = new Circle(5.0);  // OK
```

### 6.4 Operator Overloading

Redefines how operators behave for a class.

```cpp
class Vector2D {
public:
    double x, y;
    Vector2D(double x, double y) : x(x), y(y) {}

    // Member function form
    Vector2D operator+(const Vector2D& other) const {
        return Vector2D(x + other.x, y + other.y);
    }

    // Comparison
    bool operator==(const Vector2D& other) const {
        return x == other.x && y == other.y;
    }
};

// Global function form (useful when left operand is not the class)
ostream& operator<<(ostream& os, const Vector2D& v) {
    return os << "(" << v.x << ", " << v.y << ")";
}

Vector2D a(1, 2), b(3, 4);
Vector2D c = a + b;
cout << c;  // (4, 6)
```

---

## 7. Templates

### 7.1 Function Template

Write one function that works for any type.

```cpp
template <typename T>
T maxOf(T a, T b) {
    return (a > b) ? a : b;
}

cout << maxOf(3, 5);        // 5 (int)
cout << maxOf(1.2, 0.8);    // 1.2 (double)
```

### 7.2 Explicit Specialization

Override the template for a specific type.

```cpp
template <typename T>
void print(T val) { cout << val; }

template <>
void print<bool>(bool val) { cout << (val ? "true" : "false"); }

print(42);    // 42
print(true);  // true
```

### 7.3 Class Template

```cpp
template <typename T>
class Stack {
    vector<T> data;
public:
    void push(T val) { data.push_back(val); }
    T pop() { T top = data.back(); data.pop_back(); return top; }
    bool empty() { return data.empty(); }
};

Stack<int> si;
Stack<string> ss;
si.push(1); si.push(2);
cout << si.pop();  // 2
```

Default type parameter:

```cpp
template <typename T = int>
class Container { ... };

Container<> c;       // T = int
Container<double> d;
```

---

## 8. Smart Pointers

Smart pointers manage dynamic memory automatically, preventing leaks.
Include `<memory>`.

### 8.1 `unique_ptr`

Sole owner — cannot be copied, only moved.

```cpp
unique_ptr<int> p1 = make_unique<int>(10);
cout << *p1;              // 10

unique_ptr<int> p2 = move(p1);  // ownership transferred
// p1 is now nullptr

p2.reset();               // explicitly release memory
```

Array version:

```cpp
unique_ptr<int[]> arr = make_unique<int[]>(5);
arr[0] = 42;
```

### 8.2 `shared_ptr` & `weak_ptr`

`shared_ptr`: multiple owners, reference-counted. Memory freed when count reaches 0.

```cpp
shared_ptr<int> sp1 = make_shared<int>(42);
shared_ptr<int> sp2 = sp1;        // count = 2
cout << sp1.use_count();          // 2
sp2.reset();                      // count = 1
// sp1 goes out of scope → count = 0 → memory freed
```

`weak_ptr`: observes a `shared_ptr` without owning it. Breaks circular reference cycles.

```cpp
weak_ptr<int> wp = sp1;
if (auto locked = wp.lock()) {   // lock() returns shared_ptr if still alive
    cout << *locked;
}
```

> Circular `shared_ptr` references cause memory leaks — use `weak_ptr` to break the cycle.

---

## 9. STL Basics

### 9.1 Stack

LIFO (Last In, First Out). Include `<stack>`.

```cpp
stack<int> s;
s.push(1); s.push(5); s.push(2);
cout << s.top();   // 2
s.pop();
cout << s.empty(); // false
cout << s.size();  // 2
```

### 9.2 Queue

FIFO (First In, First Out). Include `<queue>`.

```cpp
queue<int> q;
q.push(1); q.push(2); q.push(3);
cout << q.front();  // 1
cout << q.back();   // 3
q.pop();
```

### 9.3 Vector

Dynamic resizable array. Include `<vector>`.

```cpp
vector<int> v = {1, 2, 3};
v.push_back(4);           // append
v.pop_back();             // remove last
cout << v[0];             // access by index
cout << v.size();         // element count
v.erase(v.begin() + 1);  // remove element at index 1
v.insert(v.begin(), 0);  // insert 0 at front
v.clear();                // remove all

// Iteration
for (int x : v) cout << x;
for (int i = 0; i < v.size(); i++) cout << v[i];
```
