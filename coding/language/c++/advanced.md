# C++ Advanced

## Table of Contents

1. [Modern C++ Syntax](#1-modern-c-syntax)
   - 1.1 [`auto` and Type Deduction](#11-auto-and-type-deduction)
   - 1.2 [`nullptr`](#12-nullptr)
   - 1.3 [`constexpr`](#13-constexpr)
   - 1.4 [Structured Bindings (C++17)](#14-structured-bindings-c17)
2. [Move Semantics](#2-move-semantics)
   - 2.1 [Lvalue vs Rvalue](#21-lvalue-vs-rvalue)
   - 2.2 [Move Constructor & Move Assignment](#22-move-constructor--move-assignment)
   - 2.3 [`std::move` vs `std::forward`](#23-stdmove-vs-stdforward)
3. [Lambda Expressions](#3-lambda-expressions)
   - 3.1 [Syntax & Capture List](#31-syntax--capture-list)
   - 3.2 [Lambdas with STL](#32-lambdas-with-stl)
4. [Callable Objects & Functional Programming](#4-callable-objects--functional-programming)
   - 4.1 [Function Pointers](#41-function-pointers)
   - 4.2 [`std::function`](#42-stdfunction)
   - 4.3 [`std::bind`](#43-stdbind)
5. [`const` Correctness](#5-const-correctness)
6. [Class Design Patterns](#6-class-design-patterns)
   - 6.1 [`override` and `final`](#61-override-and-final)
   - 6.2 [Deleted and Defaulted Functions](#62-deleted-and-defaulted-functions)
   - 6.3 [RAII](#63-raii-resource-acquisition-is-initialization)
7. [STL Containers (Full)](#7-stl-containers-full)
   - 7.1 [`deque`](#71-deque)
   - 7.2 [`priority_queue`](#72-priority_queue)
   - 7.3 [`map` and `unordered_map`](#73-map-and-unordered_map)
   - 7.4 [`set` and `unordered_set`](#74-set-and-unordered_set)
8. [STL Algorithms](#8-stl-algorithms)
9. [Advanced Templates](#9-advanced-templates)
   - 9.1 [Variadic Templates](#91-variadic-templates)
   - 9.2 [SFINAE & `enable_if`](#92-sfinae--enable_if)
10. [Type Casting](#10-type-casting)
11. [Concurrency Basics](#11-concurrency-basics)
    - 11.1 [`thread`](#111-thread)
    - 11.2 [`mutex` and `lock_guard`](#112-mutex-and-lock_guard)
    - 11.3 [`atomic`](#113-atomic)

---

## 1. Modern C++ Syntax

### 1.1 `auto` and Type Deduction

`auto` lets the compiler deduce the type from the initializer. Reduces verbosity, especially with iterators and templates.

```cpp
auto x = 42;              // int
auto d = 3.14;            // double
auto s = string("hi");    // string
auto it = v.begin();      // vector<int>::iterator

// Always prefer auto with range-based for to avoid copies
for (auto& item : container) { ... }
```

`decltype` deduces the type of an expression without evaluating it.

```cpp
int a = 5;
decltype(a) b = 10;    // b is int

auto add(int x, int y) -> decltype(x + y) { return x + y; }
```

### 1.2 `nullptr`

Type-safe null pointer constant introduced in C++11.

```cpp
int* p = nullptr;          // prefer over NULL or 0

void foo(int x)   { cout << "int"; }
void foo(int* p)  { cout << "pointer"; }

foo(NULL);     // ambiguous — could match either overload
foo(nullptr);  // unambiguous — always matches pointer overload
```

### 1.3 `constexpr`

Evaluated at **compile time**, enabling zero-cost computation.

```cpp
constexpr int square(int x) { return x * x; }
constexpr int SIZE = square(8);  // 64, computed at compile time

// constexpr vs const:
// const     — value is immutable at runtime
// constexpr — value must be known at compile time
constexpr int a = 10;   // compile-time constant
const int b = getValue(); // runtime constant — OK, but not constexpr
```

### 1.4 Structured Bindings (C++17)

Unpack tuples, pairs, and structs into named variables.

```cpp
pair<int, string> p = {1, "Alice"};
auto [id, name] = p;         // id = 1, name = "Alice"

// Most useful in range-based for with maps
map<string, int> scores;
for (auto& [key, val] : scores)
    cout << key << ": " << val << endl;

// Works with arrays too
int arr[3] = {1, 2, 3};
auto [a, b, c] = arr;
```

---

## 2. Move Semantics

### 2.1 Lvalue vs Rvalue

| Category  | Description                        | Example          |
|-----------|------------------------------------|------------------|
| lvalue    | Has a name / persistent address    | `int x = 5;` → `x` |
| rvalue    | Temporary, no persistent address   | `5`, `x + 1`, `func()` |
| rvalue ref (`&&`) | Binds to rvalues            | `int&& r = 5;`   |

### 2.2 Move Constructor & Move Assignment

Instead of copying all data (expensive), steal resources from a temporary object.

```cpp
class Buffer {
public:
    int* data;
    size_t size;

    Buffer(size_t n) : size(n), data(new int[n]) {}

    // Copy constructor — allocates new memory and copies
    Buffer(const Buffer& other) : size(other.size), data(new int[other.size]) {
        copy(other.data, other.data + size, data);
    }

    // Move constructor — steals the pointer, leaves source empty
    Buffer(Buffer&& other) noexcept : size(other.size), data(other.data) {
        other.data = nullptr;
        other.size = 0;
    }

    // Move assignment operator
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }

    ~Buffer() { delete[] data; }
};

Buffer a(100);
Buffer b = move(a);  // move constructor — a.data is now nullptr
```

- Mark move operations `noexcept` for STL container compatibility.
- STL containers use move semantics automatically when resizing.

### 2.3 `std::move` vs `std::forward`

| Function           | Behavior                                             |
|--------------------|------------------------------------------------------|
| `std::move(x)`     | Unconditionally casts `x` to rvalue                  |
| `std::forward<T>(x)` | Preserves original value category (perfect forwarding) |

```cpp
// std::move — use when you intentionally want to transfer ownership
unique_ptr<int> p1 = make_unique<int>(10);
unique_ptr<int> p2 = move(p1);  // p1 becomes nullptr

// std::forward — use in template functions to forward args as-is
template <typename T>
void wrapper(T&& arg) {
    process(forward<T>(arg));  // lvalue stays lvalue, rvalue stays rvalue
}
```

---

## 3. Lambda Expressions

### 3.1 Syntax & Capture List

```
[capture](parameters) -> return_type { body }
```

```cpp
auto add = [](int a, int b) -> int { return a + b; };
cout << add(3, 4);  // 7

// Return type can be omitted if deducible
auto square = [](int x) { return x * x; };
```

Capture list controls access to outer variables:

```cpp
int x = 10, y = 20;

auto f1 = [x]()   { cout << x; };       // capture x by value (copy)
auto f2 = [&x]()  { x += 1; };          // capture x by reference
auto f3 = [=]()   { cout << x + y; };   // capture all by value
auto f4 = [&]()   { x++; y++; };        // capture all by reference
auto f5 = [=, &y](){ y = x; };          // mix: all by value, y by reference
```

### 3.2 Lambdas with STL

```cpp
vector<int> v = {5, 3, 1, 4, 2};

// Sort descending
sort(v.begin(), v.end(), [](int a, int b) { return a > b; });

// Find first element > 3
auto it = find_if(v.begin(), v.end(), [](int x) { return x > 3; });

// Count elements satisfying a condition
int cnt = count_if(v.begin(), v.end(), [](int x) { return x % 2 == 0; });

// Transform each element
transform(v.begin(), v.end(), v.begin(), [](int x) { return x * 2; });

// Capture external variable
int threshold = 3;
auto gt = [threshold](int x) { return x > threshold; };
```

---

## 4. Callable Objects & Functional Programming

### 4.1 Function Pointers

```cpp
int add(int a, int b) { return a + b; }
int sub(int a, int b) { return a - b; }

int (*fp)(int, int) = add;
cout << fp(3, 4);   // 7
fp = sub;
cout << fp(3, 4);   // -1
```

### 4.2 `std::function`

A general-purpose wrapper that can hold any callable: functions, lambdas, functors.
Include `<functional>`.

```cpp
#include <functional>

function<int(int, int)> f;

f = add;                                     // regular function
f = [](int a, int b) { return a * b; };     // lambda
f = multiplier(3);                           // functor / bind result

cout << f(2, 5);  // 10
```

### 4.3 `std::bind`

Partially applies arguments to a function, producing a new callable.

```cpp
#include <functional>

int multiply(int a, int b) { return a * b; }

auto triple = bind(multiply, placeholders::_1, 3);  // b = 3 fixed
cout << triple(5);   // 15
cout << triple(10);  // 30

// Reorder arguments
auto flipped = bind(multiply, placeholders::_2, placeholders::_1);
cout << flipped(2, 10);  // 20 (10 * 2)
```

> In modern C++, lambdas are generally preferred over `std::bind` for clarity.

---

## 5. `const` Correctness

`const` communicates intent and catches bugs at compile time. Use it as widely as possible.

```cpp
// const parameter — function won't modify the argument
void print(const string& s) { cout << s; }

// const method — won't modify any member variable
class Circle {
    double radius;
public:
    double area() const { return 3.14 * radius * radius; }  // safe to call on const objects
    void resize(double r) { radius = r; }                   // non-const — modifies state
};

// const object — can only call const methods
const Circle c(5.0);
c.area();    // OK
c.resize(6); // error — cannot call non-const method on const object

// Top-level const vs pointer const
const int* p1 = &x;    // pointer to const int — can't change *p1
int* const p2 = &x;    // const pointer to int — can't change p2 itself
const int* const p3 = &x;  // both const
```

---

## 6. Class Design Patterns

### 6.1 `override` and `final`

`override` catches typos and signature mismatches at compile time.
`final` prevents further overriding.

```cpp
class Base {
public:
    virtual void render() {}
    virtual void update() {}
};

class Derived : public Base {
public:
    void render() override {}   // OK — compiler verifies Base::render exists
    void updat() override {}    // error — typo caught at compile time
};

class Leaf : public Derived {
public:
    void render() final {}      // cannot be overridden in further subclasses
};
```

### 6.2 Deleted and Defaulted Functions

Explicitly control which special member functions exist.

```cpp
class Singleton {
public:
    Singleton(const Singleton&) = delete;             // no copy
    Singleton& operator=(const Singleton&) = delete;  // no copy assignment
    Singleton(Singleton&&) = delete;                  // no move
    Singleton() = default;                            // explicit default constructor
};

// Useful for enforcing ownership semantics:
class UniqueResource {
public:
    UniqueResource() = default;
    UniqueResource(const UniqueResource&) = delete;   // non-copyable
    UniqueResource(UniqueResource&&) = default;        // movable
};
```

### 6.3 RAII (Resource Acquisition Is Initialization)

Tie resource lifetime to object lifetime. The destructor guarantees cleanup even when exceptions occur.

```cpp
class FileHandle {
    FILE* f;
public:
    FileHandle(const char* path, const char* mode) {
        f = fopen(path, mode);
        if (!f) throw runtime_error("cannot open file");
    }
    ~FileHandle() { if (f) fclose(f); }  // guaranteed cleanup

    // Prevent copying — file handle shouldn't be duplicated
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;
};

void process() {
    FileHandle fh("data.txt", "r");  // file opened
    // ... use fh ...
}  // fh destroyed here — file automatically closed, even if exception thrown
```

Smart pointers (`unique_ptr`, `shared_ptr`) are the standard RAII tool for heap memory.

---

## 7. STL Containers (Full)

### 7.1 `deque`

Double-ended queue. O(1) push/pop at both ends; slower random access than `vector`.

```cpp
#include <deque>
deque<int> dq = {2, 3, 4};
dq.push_front(1);   // {1, 2, 3, 4}
dq.push_back(5);    // {1, 2, 3, 4, 5}
dq.pop_front();     // {2, 3, 4, 5}
cout << dq[1];      // 3
```

### 7.2 `priority_queue`

Heap-based container. Top element is always the largest (max-heap by default).

```cpp
#include <queue>

// Max-heap (default)
priority_queue<int> pq;
pq.push(3); pq.push(1); pq.push(5);
cout << pq.top();  // 5
pq.pop();
cout << pq.top();  // 3

// Min-heap
priority_queue<int, vector<int>, greater<int>> minPq;
minPq.push(3); minPq.push(1); minPq.push(5);
cout << minPq.top();  // 1

// Custom comparator with struct
struct Task { int priority; string name; };
auto cmp = [](Task a, Task b) { return a.priority < b.priority; };
priority_queue<Task, vector<Task>, decltype(cmp)> taskQueue(cmp);
```

### 7.3 `map` and `unordered_map`

| Feature          | `map`              | `unordered_map`     |
|------------------|--------------------|---------------------|
| Ordering         | Sorted by key      | No order            |
| Lookup           | O(log n)           | O(1) average        |
| Implementation   | Red-black tree     | Hash table          |
| Key requirement  | Comparable (`<`)   | Hashable            |

```cpp
#include <map>
map<string, int> m;
m["alice"] = 90;
m["bob"]   = 85;
m["carol"] = 95;

// Iteration is in sorted key order
for (auto& [key, val] : m)
    cout << key << ": " << val << "\n";

// Safe access
if (m.count("alice"))     cout << m["alice"];   // check before access
if (m.contains("bob"))    cout << m["bob"];      // C++20

m.erase("bob");

#include <unordered_map>
unordered_map<string, int> um;
um["x"] = 1;
um.reserve(100);  // pre-allocate buckets to reduce rehashing
```

### 7.4 `set` and `unordered_set`

| Feature       | `set`          | `unordered_set`  |
|---------------|----------------|------------------|
| Ordering      | Sorted         | No order         |
| Lookup        | O(log n)       | O(1) average     |
| Duplicates    | Not allowed    | Not allowed      |

```cpp
#include <set>
set<int> s = {3, 1, 4, 1, 5};  // {1, 3, 4, 5} — sorted, no duplicates
s.insert(2);
s.erase(3);
if (s.count(4)) cout << "found";

// Iterate in sorted order
for (int x : s) cout << x << " ";  // 1 2 4 5

#include <unordered_set>
unordered_set<int> us = {3, 1, 4, 1};  // {1, 3, 4} — no order guaranteed
us.insert(5);
```

---

## 8. STL Algorithms

Include `<algorithm>`. All algorithms work with iterators and accept custom comparators or predicates.

```cpp
vector<int> v = {5, 3, 1, 4, 2};

// Sorting
sort(v.begin(), v.end());                         // ascending: {1,2,3,4,5}
sort(v.begin(), v.end(), greater<int>());         // descending: {5,4,3,2,1}
stable_sort(v.begin(), v.end());                  // stable (preserves equal-element order)

// Searching
auto it  = find(v.begin(), v.end(), 3);           // iterator to 3, or v.end()
auto it2 = find_if(v.begin(), v.end(), [](int x){ return x > 3; });
bool has = binary_search(v.begin(), v.end(), 3);  // requires sorted range

// Counting
int cnt = count(v.begin(), v.end(), 1);
int cnt2 = count_if(v.begin(), v.end(), [](int x){ return x % 2 == 0; });

// Modifying
reverse(v.begin(), v.end());
transform(v.begin(), v.end(), v.begin(), [](int x){ return x * 2; });
fill(v.begin(), v.end(), 0);
replace(v.begin(), v.end(), 1, 99);               // replace all 1s with 99
remove(v.begin(), v.end(), 0);                    // moves zeros to end, returns new end

// Min / Max
int mx = *max_element(v.begin(), v.end());
int mn = *min_element(v.begin(), v.end());
auto [lo, hi] = minmax_element(v.begin(), v.end());

// Accumulate (in <numeric>)
#include <numeric>
int sum = accumulate(v.begin(), v.end(), 0);      // 0 is initial value
```

---

## 9. Advanced Templates

### 9.1 Variadic Templates

Accept any number of type arguments.

```cpp
// Base case (empty)
void print() {}

// Recursive case
template <typename T, typename... Args>
void print(T first, Args... rest) {
    cout << first;
    if constexpr (sizeof...(rest) > 0) cout << ", ";
    print(rest...);
}

print(1, "hello", 3.14, true);  // 1, hello, 3.14, 1

// Fold expression (C++17) — more concise
template <typename... Args>
auto sum(Args... args) { return (args + ...); }  // expands to a + b + c + ...
cout << sum(1, 2, 3, 4);  // 10
```

### 9.2 SFINAE & `enable_if`

Selectively enable a template only for certain types.

```cpp
#include <type_traits>

// Only enabled for integral types
template <typename T, typename = enable_if_t<is_integral_v<T>>>
void printInt(T val) { cout << "int: " << val; }

// Only enabled for floating-point types
template <typename T, typename = enable_if_t<is_floating_point_v<T>>>
void printFloat(T val) { cout << "float: " << val; }

printInt(42);      // OK
printInt(3.14);    // error — disabled for double

// C++20 alternative using concepts (cleaner)
template <integral T>
void printInt(T val) { cout << val; }
```

---

## 10. Type Casting

Prefer C++ casts over C-style `(Type)value` — they are safer and more explicit.

| Cast                    | When to Use                                                 |
|-------------------------|-------------------------------------------------------------|
| `static_cast<T>`        | Compile-time safe conversions (numeric, up/down-cast)       |
| `dynamic_cast<T>`       | Safe runtime downcast in a polymorphic hierarchy            |
| `const_cast<T>`         | Add or remove `const` (use sparingly)                       |
| `reinterpret_cast<T>`   | Bit-level reinterpretation (very low-level, avoid if possible)|

```cpp
// static_cast
double d = 3.99;
int i = static_cast<int>(d);     // 3 — truncates, no runtime cost

// dynamic_cast (requires virtual function in base)
Animal* a = new Dog();
Dog* dog = dynamic_cast<Dog*>(a);  // OK — Dog* returned
Cat* cat = dynamic_cast<Cat*>(a);  // returns nullptr — safe failure
if (dog) dog->bark();

// const_cast (rarely needed — usually indicates a design issue)
const char* cs = "hello";
char* s = const_cast<char*>(cs);  // removes const — UB if you actually write to it

// reinterpret_cast (hardware-level / network byte manipulation)
int x = 0x41424344;
char* bytes = reinterpret_cast<char*>(&x);
```

---

## 11. Concurrency Basics

### 11.1 `thread`

Include `<thread>`.

```cpp
#include <thread>

void task(int id) {
    cout << "thread " << id << " running\n";
}

int main() {
    thread t1(task, 1);
    thread t2(task, 2);

    t1.join();  // wait for t1 to finish
    t2.join();  // wait for t2 to finish

    // Detach — thread runs independently, not waited on
    thread t3(task, 3);
    t3.detach();

    return 0;
}
```

### 11.2 `mutex` and `lock_guard`

Prevent data races when multiple threads access shared state.

```cpp
#include <thread>
#include <mutex>

mutex mtx;
int counter = 0;

void increment() {
    for (int i = 0; i < 1000; i++) {
        lock_guard<mutex> lock(mtx);  // locks on construction, unlocks on destruction
        counter++;
    }
}

int main() {
    thread t1(increment), t2(increment);
    t1.join(); t2.join();
    cout << counter;  // always 2000
}
```

`unique_lock` offers more flexibility (e.g., manual unlock, use with condition variables):

```cpp
unique_lock<mutex> lock(mtx);
// ... critical section ...
lock.unlock();   // can unlock early if needed
```

### 11.3 `atomic`

For simple shared variables, `atomic` avoids the overhead of a mutex.

```cpp
#include <atomic>

atomic<int> counter(0);

void increment() {
    for (int i = 0; i < 1000; i++)
        counter++;   // thread-safe, no mutex needed
}
```

| Tool            | Use Case                                          |
|-----------------|---------------------------------------------------|
| `atomic<T>`     | Single variable, simple read/write/increment      |
| `mutex`         | Protecting a block of code or multiple variables  |
| `lock_guard`    | Scoped mutex lock — always prefer over manual lock/unlock |
| `unique_lock`   | Flexible mutex lock — needed for condition variables |
