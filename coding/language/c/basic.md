# C Basic

## Table of Contents

1. [Program Structure](#1-program-structure)
   - 1.1 [Hello World & I/O](#11-hello-world--io)
   - 1.2 [Preprocessor Directives](#12-preprocessor-directives)
2. [Memory Model](#2-memory-model)
   - 2.1 [Memory Segments](#21-memory-segments)
   - 2.2 [Variable Types by Storage](#22-variable-types-by-storage)
3. [Pointers](#3-pointers)
   - 3.1 [Basic Pointers](#31-basic-pointers)
   - 3.2 [Double Pointers](#32-double-pointers)
   - 3.3 [Pointers and Arrays](#33-pointers-and-arrays)
4. [Functions & Parameters](#4-functions--parameters)
   - 4.1 [Pass by Value](#41-pass-by-value)
   - 4.2 [Pass by Reference (via Pointer)](#42-pass-by-reference-via-pointer)
5. [Dynamic Memory Allocation](#5-dynamic-memory-allocation)
6. [Structs](#6-structs)
   - 6.1 [Basic Struct](#61-basic-struct)
   - 6.2 [typedef](#62-typedef)
   - 6.3 [Struct with Pointer (`->`)](#63-struct-with-pointer)
7. [Preprocessor](#7-preprocessor)
   - 7.1 [`#include`](#71-include)
   - 7.2 [`#define`](#72-define)
   - 7.3 [Conditional Compilation (`#ifndef` / `#endif`)](#73-conditional-compilation-ifndef--endif)
   - 7.4 [Modular Compilation](#74-modular-compilation)

---

## 1. Program Structure

### 1.1 Hello World & I/O

```c
#include <stdio.h>

int main() {
    printf("Hello, World!\n");

    int x;
    scanf("%d", &x);       // read integer from stdin
    printf("%d\n", x);

    return 0;
}
```

Common format specifiers:

| Specifier | Type     |
|-----------|----------|
| `%d`      | int      |
| `%f`      | float / double |
| `%c`      | char     |
| `%s`      | string (char array) |
| `%p`      | pointer address |

### 1.2 Preprocessor Directives

Processed before compilation. See [Section 7](#7-preprocessor) for full details.

```c
#include <stdio.h>   // standard library header
#define MAX 100      // compile-time constant
```

---

## 2. Memory Model

### 2.1 Memory Segments

The OS divides program memory into four segments:

| Segment     | Contents                                      |
|-------------|-----------------------------------------------|
| Code        | Compiled source code (read-only)              |
| Data        | Global variables, static variables            |
| Heap        | Dynamically allocated variables (`malloc`)    |
| Stack       | Local variables, function parameters          |

```
High address
┌──────────────┐
│    Stack     │  ← local variables, grows downward
├──────────────┤
│      ↓       │
│              │
│      ↑       │
├──────────────┤
│     Heap     │  ← malloc, grows upward
├──────────────┤
│     Data     │  ← global & static variables
├──────────────┤
│     Code     │  ← machine instructions
└──────────────┘
Low address
```

### 2.2 Variable Types by Storage

**Local variable** — lives on the stack, exists only within its scope.

```c
void foo() {
    int x = 5;  // created on call, destroyed on return
}
```

**Global variable** — lives in the data segment, exists for the entire program lifetime.

```c
int count = 0;  // accessible from any function

void increment() { count++; }
```

**Static variable** — lives in the data segment; local in scope but persists across calls.

```c
#include <stdio.h>

void process() {
    static int a = 5;  // initialized only once
    a += 1;
    printf("%d\n", a);
}

int main() {
    process();  // 6
    process();  // 7
    process();  // 8
    return 0;
}
```

**Register variable** — hint to the compiler to store the variable in a CPU register for fast access. Modern compilers optimize this automatically.

```c
register int i = 0;
```

---

## 3. Pointers

### 3.1 Basic Pointers

A pointer stores a memory address. Declared with `*`, address obtained with `&`.

```c
int a = 5;
int* b = &a;   // b holds the address of a

printf("%d\n",  a);   // 5         — value of a
printf("%p\n",  b);   // 0xAFB03954 — address stored in b
printf("%d\n", *b);   // 5         — value at that address (dereference)
printf("%p\n", &b);   // 0xCA29839F — address of b itself

*b = 10;   // modifies a through the pointer
printf("%d\n", a);    // 10
```

### 3.2 Double Pointers

A pointer to a pointer. Each level of `*` dereferences one address.

```c
int  a  = 5;
int* b  = &a;   // b  → a
int** c = &b;   // c  → b → a
```

| Expression | Meaning            | Example Value  |
|------------|--------------------|----------------|
| `&c`       | address of `c`     | `0xBC10207A`   |
| `c`        | address of `b`     | `0xCA29839F`   |
| `*c` / `b` | address of `a`     | `0xAFB03954`   |
| `**c` / `*b` / `a` | value of `a` | `5`       |

```c
**c = 99;  // modifies a through two levels of indirection
printf("%d\n", a);  // 99
```

### 3.3 Pointers and Arrays

An array variable is itself a pointer to its first element.

```c
int arr[5] = {10, 20, 30, 40, 50};
int* p = arr;       // same as &arr[0]

printf("%d\n", *p);       // 10
printf("%d\n", *(p + 1)); // 20  — pointer arithmetic
printf("%d\n", p[2]);     // 30  — array indexing syntax works on pointers too
```

---

## 4. Functions & Parameters

### 4.1 Pass by Value

A copy of the argument is made. The original variable is not changed.

```c
void add(int x) {
    x += 10;        // modifies the copy only
}

int x = 5;
add(x);
printf("%d\n", x);  // 5 — unchanged
```

### 4.2 Pass by Reference (via Pointer)

Pass the address of the variable. The function modifies the original through the pointer.

```c
void add(int* x) {
    *x += 10;       // modifies the value at the address
}

int x = 5;
add(&x);
printf("%d\n", x);  // 15
```

> C does not have a reference type (`&`) like C++. Pass-by-reference in C is always done via pointers.

---

## 5. Dynamic Memory Allocation

Allocates memory at runtime on the heap. Include `<stdlib.h>`.

```c
#include <stdlib.h>

// Basic form
Type* varname = malloc(sizeof(Type) * n);
free(varname);  // must always free to prevent memory leaks
```

```c
int n = 5;
// int arr[n];  — not allowed in standard C (VLA is optional in C11+)

int* arr = malloc(sizeof(int) * n);  // allocates space for 5 ints on the heap

arr[0] = 10;
arr[1] = 20;
// ... use arr ...

free(arr);  // release memory back to OS
arr = NULL; // good practice: prevent dangling pointer
```

**`calloc`** — like `malloc` but zero-initializes the memory:

```c
int* arr = calloc(n, sizeof(int));  // all elements set to 0
```

**`realloc`** — resize a previously allocated block:

```c
arr = realloc(arr, sizeof(int) * 10);  // expand to 10 ints
```

> Always check that `malloc` / `calloc` did not return `NULL` (allocation failure).

---

## 6. Structs

A struct groups related variables of different types under one name.

### 6.1 Basic Struct

```c
#include <stdio.h>

struct Student {
    int id;
    int score;
};

int main() {
    struct Student s = {10207, 95};  // initialize in declaration order
    printf("%d\n", s.id);            // 10207
    printf("%d\n", s.score);         // 95

    s.score = 100;                   // modify a field
    return 0;
}
```

Struct defined with an inline instance:

```c
struct Student {
    int id;
    int score;
} s1, s2;  // s1 and s2 are declared here
```

### 6.2 `typedef`

Allows declaring struct variables without the `struct` keyword.

```c
typedef struct {
    int id;
    int score;
} Student;

Student s = {10207, 95};  // no need to write 'struct Student'
```

### 6.3 Struct with Pointer

When a struct is accessed through a pointer, use `->` instead of `.`.

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int id;
    int score;
} Student;

int main() {
    Student* s = malloc(sizeof(Student));
    s->id    = 10207;   // equivalent to (*s).id = 10207
    s->score = 95;
    printf("%d %d\n", s->id, s->score);
    free(s);
    return 0;
}
```

| Access method       | When to use                       |
|---------------------|-----------------------------------|
| `s.field`           | `s` is a struct variable          |
| `s->field`          | `s` is a pointer to a struct      |

---

## 7. Preprocessor

Preprocessor directives are handled before compilation. They begin with `#` and are not C statements (no semicolon).

### 7.1 `#include`

Inserts the entire content of a file in place of the directive.

```c
#include <stdio.h>    // search in system include directories
#include "temp.h"     // search in current directory first, then system dirs
```

Example — splitting into a header and source:

```c
// temp.h
int add(int x, int y) {
    return x + y;
}

// main.c
#include <stdio.h>
#include "temp.h"   // entire content of temp.h is pasted here

int main() {
    printf("%d\n", add(10, 7));  // 17
    return 0;
}
```

> If `add` is defined in a header included multiple times, the function will be defined twice — causing a linker error. Solve this with include guards (see §7.3) or by separating declaration and definition (see §7.4).

### 7.2 `#define`

Defines a compile-time constant or macro. The preprocessor performs a text substitution before compilation.

```c
#define PI    3.14159
#define MAX   100
#define ll    long long

// Function-like macro
#define SQUARE(x) ((x) * (x))   // parentheses are important to avoid precedence bugs

double area = PI * SQUARE(5);   // expands to 3.14159 * ((5) * (5))
```

> Prefer `const` variables and `inline` functions over macros in modern C — macros have no type safety and can produce surprising results.

### 7.3 Conditional Compilation (`#ifndef` / `#endif`)

Prevents a header from being included more than once (include guard).

```c
// temp.h
#ifndef _TEMP_H_         // if _TEMP_H_ is NOT yet defined...
#define _TEMP_H_         // ...define it (so this block runs only once)

int add(int x, int y) {
    return x + y;
}

#endif                   // end of the guarded block
```

```c
// main.c
#include <stdio.h>
#include "temp.h"   // processed — _TEMP_H_ defined
#include "temp.h"   // skipped — _TEMP_H_ already defined; no error

int main() {
    printf("%d\n", add(10, 7));
    return 0;
}
```

Other conditional directives:

```c
#ifdef  SYMBOL   // true if SYMBOL is defined
#ifndef SYMBOL   // true if SYMBOL is NOT defined
#if     EXPR     // true if compile-time expression is non-zero
#elif   EXPR     // else-if branch
#else            // fallback branch
#endif           // closes the conditional block
```

### 7.4 Modular Compilation

Separate the **declaration** (header) from the **implementation** (source file). This is the standard practice for larger C projects.

```c
// temp.h — declaration only (the interface)
#ifndef _TEMP_H_
#define _TEMP_H_

int add(int x, int y);   // function prototype

#endif
```

```c
// temp.c — implementation
#include "temp.h"

int add(int x, int y) {
    return x + y;
}
```

```c
// main.c — user of the module
#include <stdio.h>
#include "temp.h"   // only needs the declaration

int main() {
    printf("%d\n", add(10, 7));  // 17
    return 0;
}
```

Compile together:

```bash
gcc main.c temp.c -o main
```

Benefits:
- Each `.c` file is compiled independently — only changed files need recompilation.
- The header exposes only the public interface; implementation details stay in `.c`.
- Include guards prevent duplicate definitions when multiple files include the same header.
