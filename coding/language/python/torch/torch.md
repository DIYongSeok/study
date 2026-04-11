# PyTorch

PyTorch is an open-source deep learning framework built on dynamic computation graphs. It is the dominant framework for research and increasingly used in production.

```bash
pip install torch torchvision
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

## Table of Contents

1. [Tensors](#1-tensors)
   - 1.1 [Creating Tensors](#11-creating-tensors)
   - 1.2 [Tensor Attributes](#12-tensor-attributes)
   - 1.3 [Indexing & Slicing](#13-indexing--slicing)
   - 1.4 [Shape Manipulation](#14-shape-manipulation)
2. [Tensor Operations](#2-tensor-operations)
   - 2.1 [Arithmetic & Broadcasting](#21-arithmetic--broadcasting)
   - 2.2 [Aggregation](#22-aggregation)
   - 2.3 [Matrix Operations](#23-matrix-operations)
3. [Autograd — Automatic Differentiation](#3-autograd--automatic-differentiation)
   - 3.1 [Computing Gradients](#31-computing-gradients)
   - 3.2 [Stopping Gradient Tracking](#32-stopping-gradient-tracking)
4. [Building Neural Networks](#4-building-neural-networks)
   - 4.1 [Defining a Model with `nn.Module`](#41-defining-a-model-with-nnmodule)
   - 4.2 [Common Layers](#42-common-layers)
   - 4.3 [Activation Functions](#43-activation-functions)
   - 4.4 [Loss Functions](#44-loss-functions)
   - 4.5 [Optimizers](#45-optimizers)
5. [Training Loop](#5-training-loop)
   - 5.1 [Full Training Template](#51-full-training-template)
   - 5.2 [Validation Loop](#52-validation-loop)
6. [Data Loading](#6-data-loading)
   - 6.1 [Dataset & DataLoader](#61-dataset--dataloader)
   - 6.2 [Custom Dataset](#62-custom-dataset)
   - 6.3 [Transforms (torchvision)](#63-transforms-torchvision)
7. [GPU Acceleration](#7-gpu-acceleration)
8. [Saving & Loading Models](#8-saving--loading-models)
9. [Useful Utilities](#9-useful-utilities)
   - 9.1 [NumPy Interoperability](#91-numpy-interoperability)
   - 9.2 [Reproducibility (Seeds)](#92-reproducibility-seeds)
   - 9.3 [Model Summary](#93-model-summary)

---

## 1. Tensors

Tensors are the fundamental data structure in PyTorch — essentially NumPy arrays that can run on GPU and support automatic differentiation.

### 1.1 Creating Tensors

```python
import torch

# From Python list
a = torch.tensor([1, 2, 3])                    # dtype inferred (int64)
b = torch.tensor([1.0, 2.0, 3.0])             # float32
c = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

# Built-in constructors
torch.zeros(3, 4)            # 3×4 tensor of 0.0
torch.ones(2, 3)             # 2×3 tensor of 1.0
torch.full((2, 3), 7.0)      # filled with 7.0
torch.eye(3)                 # 3×3 identity matrix
torch.arange(0, 10, 2)       # [0 2 4 6 8]
torch.linspace(0, 1, 5)      # [0.   0.25 0.5  0.75 1.  ]

# Random
torch.rand(3, 3)             # uniform [0, 1)
torch.randn(3, 3)            # standard normal N(0, 1)
torch.randint(0, 10, (3, 3)) # random integers in [0, 10)

# Like-constructors (same shape/dtype as an existing tensor)
x = torch.ones(2, 3)
torch.zeros_like(x)
torch.ones_like(x)
torch.rand_like(x)
```

### 1.2 Tensor Attributes

```python
t = torch.randn(2, 3)

print(t.shape)     # torch.Size([2, 3])
print(t.size())    # torch.Size([2, 3])  — same as .shape
print(t.ndim)      # 2
print(t.numel())   # 6   — total number of elements
print(t.dtype)     # torch.float32
print(t.device)    # cpu  (or cuda:0)
```

### 1.3 Indexing & Slicing

Same rules as NumPy.

```python
t = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]], dtype=torch.float32)

print(t[0])          # tensor([1., 2., 3.])   — row 0
print(t[:, 1])       # tensor([2., 5., 8.])   — column 1
print(t[0:2, 1:3])   # tensor([[2., 3.], [5., 6.]])
print(t[1, 2])       # tensor(6.)
print(t[1, 2].item())  # 6.0  — Python scalar

# Boolean indexing
mask = t > 5
print(t[mask])       # tensor([6., 7., 8., 9.])
```

### 1.4 Shape Manipulation

#### reshape

Reads all elements in order, then fills them into the new shape.

```python
t = torch.arange(12, dtype=torch.float32)
# tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.])
# shape: (12,)

t.reshape(3, 4)
# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.]])
# shape: (3, 4)

t.reshape(2, 2, 3)
# tensor([[[ 0.,  1.,  2.],
#          [ 3.,  4.,  5.]],
#         [[ 6.,  7.,  8.],
#          [ 9., 10., 11.]]])
# shape: (2, 2, 3)

t.reshape(3, -1)     # -1 infers the missing dim: 12/3 = 4 → (3, 4)
t.view(3, 4)         # same result, but requires contiguous memory
```

#### view vs reshape — contiguous memory

Both produce the same shape, but they work differently under the hood.

**Contiguous** means tensor elements are stored in one unbroken block of memory, in row-major order (left-to-right, top-to-bottom).

```python
t = torch.arange(12, dtype=torch.float32)   # freshly created → contiguous
t.is_contiguous()   # True

t.view(3, 4)        # works fine — just changes how the block is interpreted
t.reshape(3, 4)     # also works fine
```

Operations like `.transpose()` or `.permute()` do **not** move data in memory — they just change the stride (the step size used to navigate the data). The result is **non-contiguous**.

```python
m = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])
# Memory layout: [1, 2, 3, 4, 5, 6]  (row by row)

t2 = m.T             # transpose — shape (3, 2)
t2.is_contiguous()   # False ← data is still [1,2,3,4,5,6] but strides are reversed

t2.view(6)           # RuntimeError: view size is not compatible with contiguous memory
t2.reshape(6)        # works — reshape makes a copy when needed
```

To use `.view()` on a non-contiguous tensor, call `.contiguous()` first:

```python
t2.contiguous().view(6)
# tensor([1., 4., 2., 5., 3., 6.])  ← copies data into a new contiguous block, then views it
```

**Summary:**

| | `view` | `reshape` |
|---|---|---|
| Requires contiguous | Yes — errors otherwise | No — copies if needed |
| Copies data | Never | Only when necessary |
| Use when | You want zero-copy guarantee | You just want the new shape |

```python
# Rule of thumb:
# - use reshape when you just want the shape and don't care about memory
# - use view when you explicitly want to confirm no copy was made
```

---

#### unsqueeze / squeeze

`unsqueeze` adds a size-1 dimension. `squeeze` removes it.

```python
t = torch.arange(4, dtype=torch.float32)
# tensor([0., 1., 2., 3.])
# shape: (4,)

t.unsqueeze(0)
# tensor([[0., 1., 2., 3.]])
# shape: (1, 4)  ← new dim at position 0 (adds a "row" wrapper)

t.unsqueeze(1)
# tensor([[0.],
#         [1.],
#         [2.],
#         [3.]])
# shape: (4, 1)  ← new dim at position 1 (adds a "column" wrapper)

x = t.unsqueeze(0)   # shape (1, 4)
x.squeeze()
# tensor([0., 1., 2., 3.])
# shape: (4,)  ← size-1 dims removed
```

---

#### transpose / permute

`T` and `transpose` swap two axes. `permute` reorders all axes at once.

```python
m = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])
# shape: (2, 3)

m.T
# tensor([[1., 4.],
#         [2., 5.],
#         [3., 6.]])
# shape: (3, 2)  ← rows become columns

t3 = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
# shape: (2, 3, 4)  — think: (batch, height, width)

t3.permute(2, 0, 1)
# shape: (4, 2, 3)  — dim2 → dim0, dim0 → dim1, dim1 → dim2
# t3[i, j, k] is now at result[k, i, j]
```

---

#### cat / stack

`cat` joins tensors along an **existing** axis. `stack` creates a **new** axis.

```python
a = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])   # shape (2, 3)

b = torch.tensor([[7., 8., 9.],
                  [0., 1., 2.]])   # shape (2, 3)

torch.cat([a, b], dim=0)
# tensor([[1., 2., 3.],
#         [4., 5., 6.],
#         [7., 8., 9.],
#         [0., 1., 2.]])
# shape: (4, 3)  ← stacked row-wise

torch.cat([a, b], dim=1)
# tensor([[1., 2., 3., 7., 8., 9.],
#         [4., 5., 6., 0., 1., 2.]])
# shape: (2, 6)  ← stacked column-wise

torch.stack([a, b], dim=0)
# tensor([[[1., 2., 3.],
#          [4., 5., 6.]],
#         [[7., 8., 9.],
#          [0., 1., 2.]]])
# shape: (2, 2, 3)  ← new dim at position 0 wraps both tensors
```

---

## 2. Tensor Operations

### 2.1 Arithmetic & Broadcasting

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

a + b           # tensor([5., 7., 9.])
a - b           # tensor([-3., -3., -3.])
a * b           # tensor([4., 10., 18.])
a / b           # tensor([0.25, 0.40, 0.50])
a ** 2          # tensor([1., 4., 9.])

# In-place operations (modifies tensor directly, saves memory)
a.add_(1)       # a += 1
a.mul_(2)       # a *= 2

# Broadcasting — same rules as NumPy
A = torch.ones(3, 4)
v = torch.arange(4, dtype=torch.float32)  # shape (4,) → broadcast to (3, 4)
print(A + v)
```

### 2.2 Aggregation

```python
t = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

t.sum()              # tensor(21.)
t.sum(dim=0)         # tensor([5., 7., 9.])   — collapse rows
t.sum(dim=1)         # tensor([6., 15.])      — collapse columns
t.mean()             # tensor(3.5)
t.max()              # tensor(6.)
t.min()              # tensor(1.)
t.std()              # standard deviation
t.argmax()           # index of global max
t.argmax(dim=1)      # index of max per row
```

### 2.3 Matrix Operations

```python
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

# Matrix multiplication
A @ B                          # tensor([[19., 22.], [43., 50.]])
torch.matmul(A, B)             # same
torch.mm(A, B)                 # same, 2D only

# Batched matrix multiply (3D tensors)
batch_A = torch.randn(32, 3, 4)
batch_B = torch.randn(32, 4, 5)
torch.bmm(batch_A, batch_B)    # (32, 3, 5)

# Element-wise multiply (NOT matrix multiply)
A * B

# Transpose
A.T
A.transpose(0, 1)              # same

# Dot product (1D vectors)
torch.dot(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]))  # 11.
```

---

## 3. Autograd — Automatic Differentiation

PyTorch automatically computes gradients through a dynamic computation graph. This is the core mechanism behind backpropagation.

### 3.1 Computing Gradients

```python
# requires_grad=True tells PyTorch to track operations on this tensor
x = torch.tensor(3.0, requires_grad=True)

# Forward pass — build computation graph
y = x ** 2 + 2 * x + 1   # y = x² + 2x + 1

# Backward pass — compute gradients (dy/dx = 2x + 2)
y.backward()

print(x.grad)   # tensor(8.)  →  2*3 + 2 = 8
```

Multi-variable example:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()   # y = x1² + x2² + x3²

y.backward()
print(x.grad)        # tensor([2., 4., 6.])  → dy/dxi = 2xi
```

**Important:** Gradients accumulate. Call `optimizer.zero_grad()` (or `x.grad.zero_()`) before each backward pass.

### 3.2 Stopping Gradient Tracking

```python
x = torch.randn(3, requires_grad=True)

# Context manager — preferred
with torch.no_grad():
    y = x * 2   # no gradient computation

# Decorator
@torch.no_grad()
def inference(model, x):
    return model(x)

# Detach — creates a new tensor sharing data but without gradient history
z = x.detach()
print(z.requires_grad)  # False
```

---

## 4. Building Neural Networks

### 4.1 Defining a Model with `nn.Module`

All models in PyTorch are subclasses of `nn.Module`.

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()                      # always call super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):                       # define the forward pass
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MLP(input_size=784, hidden_size=256, output_size=10)
print(model)

# Inspect parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

### 4.2 [Common Layers](./layers.md)

```python
nn.Linear(in_features, out_features)           # fully connected: y = xW^T + b
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
nn.MaxPool2d(kernel_size, stride)
nn.AvgPool2d(kernel_size, stride)
nn.BatchNorm1d(num_features)                   # normalize over a mini-batch (1D)
nn.BatchNorm2d(num_features)                   # for 2D feature maps
nn.LayerNorm(normalized_shape)
nn.Dropout(p=0.5)                              # randomly zero elements with prob p
nn.Dropout2d(p=0.5)                            # zero entire channels
nn.Embedding(num_embeddings, embedding_dim)    # learnable lookup table
nn.LSTM(input_size, hidden_size, num_layers)
nn.GRU(input_size, hidden_size, num_layers)
nn.MultiheadAttention(embed_dim, num_heads)
nn.Sequential(layer1, layer2, ...)             # stack layers without subclassing
```

Using `nn.Sequential`:

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

### 4.3 [Activation Functions](./activation_functions.md)

```python
import torch.nn.functional as F

# As layers (stateless, can go in nn.Sequential)
nn.ReLU()
nn.Sigmoid()
nn.Tanh()
nn.Softmax(dim=1)
nn.LeakyReLU(negative_slope=0.01)
nn.GELU()

# As functions (used inside forward())
F.relu(x)
F.sigmoid(x)
F.tanh(x)
F.softmax(x, dim=1)
F.gelu(x)
```

### 4.4 [Loss Functions](./loss_functions.md)

```python
# Classification
nn.CrossEntropyLoss()        # combines LogSoftmax + NLLLoss; expects raw logits
nn.BCELoss()                 # binary cross-entropy; expects probabilities (after sigmoid)
nn.BCEWithLogitsLoss()       # binary cross-entropy with built-in sigmoid; preferred

# Regression
nn.MSELoss()                 # mean squared error
nn.L1Loss()                  # mean absolute error
nn.SmoothL1Loss()            # Huber loss — less sensitive to outliers

# Usage
criterion = nn.CrossEntropyLoss()
loss = criterion(predictions, targets)  # predictions: (N, C) logits; targets: (N,) class indices
```

### 4.5 Optimizers

```python
import torch.optim as optim

# Common optimizers
optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)  # Adam + decoupled weight decay
optim.RMSprop(model.parameters(), lr=1e-3)

# Learning rate schedulers
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

# Call after each epoch
scheduler.step()
```

---

## 5. Training Loop

### 5.1 Full Training Template

```python
model = MLP(784, 256, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()                               # set to training mode (enables dropout, batchnorm)

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        # 1. Move data to device (GPU if available)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # 2. Zero gradients (prevent accumulation from previous step)
        optimizer.zero_grad()

        # 3. Forward pass
        outputs = model(batch_x)

        # 4. Compute loss
        loss = criterion(outputs, batch_y)

        # 5. Backward pass (compute gradients)
        loss.backward()

        # 6. Update weights
        optimizer.step()

        # Track metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc  = 100. * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.2f}%")
```

### 5.2 Validation Loop

```python
model.eval()                                    # disable dropout and batchnorm updates

val_loss = 0.0
correct = 0
total = 0

with torch.no_grad():                           # no gradient computation needed
    for batch_x, batch_y in val_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        val_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

val_loss /= len(val_loader)
val_acc = 100. * correct / total
print(f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")
```

Key distinction:

| `model.train()`          | `model.eval()`                  |
|--------------------------|---------------------------------|
| Dropout active           | Dropout disabled                |
| BatchNorm uses batch stats | BatchNorm uses running stats  |
| Used during training     | Used during validation/testing  |

---

## 6. Data Loading

### 6.1 Dataset & DataLoader

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True,
                                download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False,
                                download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64,
                          shuffle=True, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=64,
                          shuffle=False, num_workers=2)

# Inspect one batch
batch_x, batch_y = next(iter(train_loader))
print(batch_x.shape)  # torch.Size([64, 1, 28, 28])
print(batch_y.shape)  # torch.Size([64])
```

### 6.2 Custom Dataset

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data           # e.g., list of file paths or a tensor
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)      # total number of samples

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

dataset = MyDataset(data, labels)
loader  = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 6.3 Transforms (torchvision)

```python
from torchvision import transforms

# Common pipeline for training images
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),                                  # PIL → [0,1] float tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],       # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

# Inference — no augmentation
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

---

## 7. GPU Acceleration

```python
# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)           # cuda:0  or  cpu

print(torch.cuda.get_device_name(0))    # e.g., "NVIDIA GeForce RTX 3090"
print(torch.cuda.memory_allocated())    # bytes currently allocated

# Move model and data to GPU
model = model.to(device)

for x, y in loader:
    x, y = x.to(device), y.to(device)
    output = model(x)
    ...

# Move tensor between devices
t = torch.randn(3, 3)
t_gpu = t.to("cuda")     # CPU → GPU
t_cpu = t_gpu.to("cpu")  # GPU → CPU
t_cpu = t_gpu.cpu()      # shorthand
```

---

## 8. Saving & Loading Models

```python
# ── Recommended: save only the state dict (weights) ──

# Save
torch.save(model.state_dict(), "model.pth")

# Load (must create the model architecture first)
model = MLP(784, 256, 10)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# ── Save full checkpoint (weights + optimizer + epoch) ──

torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss,
}, "checkpoint.pth")

# Load full checkpoint
checkpoint = torch.load("checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"] + 1
```

---

## 9. Useful Utilities

### 9.1 NumPy Interoperability

```python
import numpy as np

# NumPy → PyTorch
arr = np.array([1.0, 2.0, 3.0])
t = torch.from_numpy(arr)      # shares memory — modifying one changes the other
t = torch.tensor(arr)          # copies data — independent

# PyTorch → NumPy  (tensor must be on CPU)
arr = t.numpy()                # shares memory
arr = t.detach().cpu().numpy() # safe version when gradients may be attached
```

### 9.2 Reproducibility (Seeds)

```python
import random, os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)              # for multi-GPU
    torch.backends.cudnn.deterministic = True     # deterministic conv algorithms
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

### 9.3 Model Summary

```python
# Option 1: print the model
print(model)

# Option 2: count parameters manually
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total:,}  Trainable: {trainable:,}")

# Option 3: torchinfo (pip install torchinfo)
from torchinfo import summary
summary(model, input_size=(1, 784))
# shows each layer's output shape, parameter count, and memory usage
```
