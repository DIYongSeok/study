# Loss Functions in PyTorch (`torch.nn`)

A loss function measures how far the model's predictions are from the true targets. The training loop minimizes this value by adjusting weights via backpropagation.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

## Table of Contents

1. [How Loss Functions Work](#1-how-loss-functions-work)
2. [Characteristics of a Good Loss Function](#2-characteristics-of-a-good-loss-function)
3. [Classification Losses](#3-classification-losses)
   - 3.1 [CrossEntropyLoss](#31-crossentropyloss)
   - 3.2 [BCELoss](#32-bceloss)
   - 3.3 [BCEWithLogitsLoss](#33-bcewithlogitsloss)
   - 3.4 [NLLLoss](#34-nllloss)
4. [Regression Losses](#4-regression-losses)
   - 4.1 [MSELoss](#41-mseloss)
   - 4.2 [L1Loss](#42-l1loss)
   - 4.3 [SmoothL1Loss (Huber)](#43-smoothl1loss-huber)
5. [Reduction Modes](#5-reduction-modes)
6. [Class Weights & Imbalanced Data](#6-class-weights--imbalanced-data)
7. [Custom Loss Functions](#7-custom-loss-functions)
8. [Quick Reference](#8-quick-reference)

---

## 1. How Loss Functions Work

```python
criterion = nn.CrossEntropyLoss()

predictions = model(x)          # forward pass → raw output (logits)
loss = criterion(predictions, targets)  # compute scalar loss

loss.backward()                  # compute gradients
optimizer.step()                 # update weights
```

The loss is always a **single scalar** — the average error across the batch. The goal of training is to make this number as small as possible.

---

## 2. Characteristics of a Good Loss Function

### 1. Differentiable everywhere (or almost everywhere)
The optimizer needs gradients to update weights. A loss that has no gradient at critical points can't be minimized.

```
ReLU has a non-differentiable point at 0 — but it's acceptable because
it only happens at one point (measure zero).

A step function (0 or 1) has zero gradient everywhere → useless as a loss.
```

### 2. Reflects the actual task goal
The loss should measure exactly what you care about.

```
Bad:  MSELoss for classification (measures distance, not correctness)
Good: CrossEntropyLoss for classification (measures probability of correct class)

Bad:  CrossEntropyLoss for bounding box regression (doesn't measure spatial distance)
Good: SmoothL1Loss for bounding box regression
```

### 3. Sensitive to the right errors
A good loss should penalize **confident wrong predictions** much more than **uncertain ones**.

```
CrossEntropyLoss:
  model says 99% → wrong class  →  loss = 4.6   (huge penalty)
  model says 50% → wrong class  →  loss = 0.69  (moderate)
  model says 99% → right class  →  loss = 0.01  (almost nothing)
```

MSELoss does this too via squaring — a large error is penalized quadratically.

### 4. Numerically stable
Avoid operations that produce `inf` or `NaN` during computation.

```python
# Unstable — exp overflows for large logits
loss = -torch.log(torch.exp(logits) / torch.exp(logits).sum())  # inf/inf = NaN

# Stable — LogSoftmax subtracts max first
loss = F.cross_entropy(logits, target)  # handles large values safely
```

### 5. Smooth and well-scaled gradients
- Gradients shouldn't vanish (→ 0) or explode (→ ∞)
- Loss values should be on a reasonable scale — a loss of `0.001` vs `10000` both cause slow or unstable training

```
SmoothL1Loss:  gradient is bounded — never jumps suddenly
L1Loss:        gradient is constant (±1) — can oscillate near the minimum
MSELoss:       gradient grows with error — can explode for large errors
```

### 6. Handles class imbalance (for classification)
A naive loss can be "minimized" by just always predicting the majority class.

```python
# 95% class 0, 5% class 1 → model always predicts class 0 → 95% accuracy, loss still low
# Fix: add class weights
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 19.0]))
# now a mistake on class 1 costs 19x more
```

| Characteristic | Why it matters |
|---|---|
| Differentiable | Gradient descent requires gradients |
| Matches the task | Wrong loss → model optimizes the wrong thing |
| Penalizes confidence | Prevents the model from being confidently wrong |
| Numerically stable | Prevents `NaN`/`inf` during training |
| Smooth gradients | Stable, steady learning |
| Handles imbalance | Prevents bias toward majority class |

The most common mistake is using a loss that is **technically valid but doesn't match what you actually want to optimize** — e.g. optimizing MSE when you care about ranking, or using accuracy directly (non-differentiable) instead of cross-entropy.

---

## 3. Classification Losses

### 3.1 CrossEntropyLoss

The standard loss for **multi-class classification** (e.g. 10 classes in MNIST).

Combines `LogSoftmax + NLLLoss` internally — pass **raw logits**, not probabilities.

```python
criterion = nn.CrossEntropyLoss()

# predictions: (N, C) — N samples, C classes, raw logits
# targets:     (N,)   — integer class indices (0 to C-1)
predictions = torch.tensor([[2.0, 1.0, 0.5],   # sample 1 logits
                             [0.5, 2.5, 0.3]])  # sample 2 logits
targets = torch.tensor([0, 1])                  # sample 1 → class 0, sample 2 → class 1

loss = criterion(predictions, targets)
print(loss)  # tensor(0.4019)
```

---

#### Step-by-step: what actually happens inside

Let's trace through **sample 1** manually: logits `[2.0, 1.0, 0.5]`, target class `0`.

**Step 1 — Softmax:** convert raw logits into probabilities (sum to 1)

```python
logits = torch.tensor([2.0, 1.0, 0.5])

softmax = torch.exp(logits) / torch.exp(logits).sum()
# exp([2.0, 1.0, 0.5]) = [7.389, 2.718, 1.649]
# sum = 11.756
# softmax = [0.6285, 0.2311, 0.1402]  ← probabilities, sum = 1.0
```

The model says: 62.8% chance class 0, 23.1% class 1, 14.0% class 2.

**Step 2 — Log:** take the natural log of each probability

```python
log_probs = torch.log(softmax)
# log([0.6285, 0.2311, 0.1402]) = [-0.4643, -1.4643, -1.9643]
```

Log turns probabilities (0 to 1) into negative numbers (−∞ to 0).
A high probability → log close to 0. A low probability → very negative log.

**Step 3 — Pick the target class score (NLLLoss):** index into log_probs at the target class

```python
target = 0
loss_sample1 = -log_probs[target]
# -(-0.4643) = 0.4643
```

The negative sign flips it: a high probability for the correct class → small loss. A low probability → large loss.

**Step 4 — Average over the batch**

```python
# sample 2: logits [0.5, 2.5, 0.3], target class 1
# softmax = [0.1173, 0.8654, 0.0966]  (high for class 1 — good prediction)
# log_probs[1] = log(0.8654) = -0.1445
# loss_sample2 = -(-0.1445) = 0.1445 (low loss — model was confident and correct)

loss = (0.4643 + 0.1445) / 2   # = 0.3044... ≈ 0.4019 (slight diff due to rounding above)
```

---

#### Visualizing why log works

```
Correct class probability → loss
        1.0  →  0.0    (perfect prediction, no loss)
        0.8  →  0.22   (good, small loss)
        0.5  →  0.69   (uncertain, moderate loss)
        0.2  →  1.61   (bad prediction, high loss)
        0.01 →  4.61   (very wrong, huge loss)
```

The loss grows **exponentially** as the model becomes more wrong — strongly penalizes confident wrong predictions.

---

#### Full manual implementation

```python
def cross_entropy_manual(logits, targets):
    # Step 1: Softmax
    exp_logits = torch.exp(logits)
    softmax = exp_logits / exp_logits.sum(dim=1, keepdim=True)

    # Step 2: Log
    log_probs = torch.log(softmax)

    # Step 3: Pick target class score (NLLLoss)
    N = logits.shape[0]
    target_log_probs = log_probs[range(N), targets]  # index each row at target column

    # Step 4: Negate and average
    loss = -target_log_probs.mean()
    return loss

predictions = torch.tensor([[2.0, 1.0, 0.5],
                             [0.5, 2.5, 0.3]])
targets = torch.tensor([0, 1])

print(cross_entropy_manual(predictions, targets))  # tensor(0.4019)
print(nn.CrossEntropyLoss()(predictions, targets)) # tensor(0.4019) ✓
```

---

#### Why not just use Softmax + BCELoss?

`CrossEntropyLoss` uses **LogSoftmax** (log + softmax fused) instead of plain Softmax.
This is more **numerically stable** — plain softmax can overflow with very large logits:

```python
# Problem with large logits
logits = torch.tensor([1000.0, 1001.0, 1002.0])
torch.exp(logits)          # tensor([inf, inf, inf]) ← overflow!

# LogSoftmax avoids this by subtracting the max first (mathematically equivalent)
F.log_softmax(logits, dim=0)  # tensor([-2.4076, -1.4076, -0.4076]) ← safe ✓
```

**Do NOT add `nn.Softmax` before this loss** — it applies softmax internally, so adding it again produces wrong gradients.

---

### 3.2 BCELoss

**Binary Cross-Entropy** for **binary classification** (output is a single probability).

Expects predictions **after sigmoid** — values in (0, 1).

```python
criterion = nn.BCELoss()

predictions = torch.tensor([0.9, 0.2, 0.8, 0.1])  # after sigmoid
targets     = torch.tensor([1.0, 0.0, 1.0, 0.0])  # binary labels

loss = criterion(predictions, targets)
print(loss)  # tensor(0.1643)
```

⚠️ Input must be in (0, 1). If you pass raw logits, you get incorrect results or NaNs.

---

### 3.3 BCEWithLogitsLoss

Same as `BCELoss` but **applies sigmoid internally**. More numerically stable — preferred in practice.

Pass **raw logits** (not sigmoid output).

```python
criterion = nn.BCEWithLogitsLoss()

predictions = torch.tensor([2.0, -1.5, 1.8, -2.0])  # raw logits
targets     = torch.tensor([1.0,  0.0, 1.0,  0.0])  # binary labels

loss = criterion(predictions, targets)
print(loss)  # tensor(0.1816)
```

| | `BCELoss` | `BCEWithLogitsLoss` |
|---|---|---|
| Input | Probabilities (after sigmoid) | Raw logits |
| Sigmoid | You apply it | Applied internally |
| Stability | Less stable | More stable ✅ |

---

### 3.4 NLLLoss

**Negative Log-Likelihood Loss** — lower-level building block used inside `CrossEntropyLoss`.

Expects **log-probabilities** as input (output of `log_softmax`).

```python
criterion = nn.NLLLoss()

log_probs = F.log_softmax(torch.tensor([[2.0, 1.0, 0.5],
                                         [0.5, 2.5, 0.3]]), dim=1)
targets = torch.tensor([0, 1])

loss = criterion(log_probs, targets)
# equivalent to nn.CrossEntropyLoss()
```

In practice, just use `CrossEntropyLoss` — it combines both steps.

---

## 4. Regression Losses

### 4.1 MSELoss

**Mean Squared Error** — averages the squared differences between predictions and targets.

**Formula:** `L = mean((ŷ - y)²)`

```python
criterion = nn.MSELoss()

predictions = torch.tensor([2.5, 0.0, 2.0, 8.0])
targets     = torch.tensor([3.0, 0.0, 2.0, 7.0])

loss = criterion(predictions, targets)
print(loss)  # tensor(0.1250)   ← (0.25 + 0 + 0 + 1) / 4
```

- Penalizes **large errors heavily** (squaring amplifies them)
- Sensitive to outliers
- Most common regression loss

---

### 4.2 L1Loss

**Mean Absolute Error** — averages the absolute differences.

**Formula:** `L = mean(|ŷ - y|)`

```python
criterion = nn.L1Loss()

predictions = torch.tensor([2.5, 0.0, 2.0, 8.0])
targets     = torch.tensor([3.0, 0.0, 2.0, 7.0])

loss = criterion(predictions, targets)
print(loss)  # tensor(0.3750)   ← (0.5 + 0 + 0 + 1) / 4
```

- Less sensitive to outliers than MSE (no squaring)
- Gradient is constant (±1) — can cause instability near zero

---

### 4.3 SmoothL1Loss (Huber)

**Huber Loss** — combines L1 and MSE. L2 (squared) for small errors, L1 (absolute) for large errors.

```python
criterion = nn.SmoothL1Loss(beta=1.0)  # beta controls the transition point

predictions = torch.tensor([2.5, 0.0, 2.0, 8.0])
targets     = torch.tensor([3.0, 0.0, 2.0, 7.0])

loss = criterion(predictions, targets)
print(loss)  # tensor(0.1250)
```

```
         MSELoss:  always squared      → large errors dominate
           L1Loss: always absolute     → robust but unstable near 0
SmoothL1Loss:      squared near 0, L1 far from 0 → best of both
```

- Best of both MSE and L1
- Widely used in **object detection** (bounding box regression)

---

## 5. Reduction Modes

All loss functions have a `reduction` parameter that controls how the per-sample losses are combined.

```python
predictions = torch.tensor([[2.0, 1.0, 0.5],
                             [0.5, 2.5, 0.3]])
targets = torch.tensor([0, 1])

nn.CrossEntropyLoss(reduction='mean')(predictions, targets)  # scalar — average (default)
nn.CrossEntropyLoss(reduction='sum')(predictions, targets)   # scalar — sum
nn.CrossEntropyLoss(reduction='none')(predictions, targets)  # tensor — one loss per sample
# tensor([0.4019, 0.1019])
```

`reduction='none'` is useful when you want to apply **sample weights** manually:

```python
criterion = nn.CrossEntropyLoss(reduction='none')
losses = criterion(predictions, targets)   # per-sample losses

weights = torch.tensor([2.0, 1.0])        # weight sample 1 more
weighted_loss = (losses * weights).mean()
```

---

## 6. Class Weights & Imbalanced Data

When one class appears far more often than others (e.g. 90% negative, 10% positive), the model ignores the rare class. Use `weight` to penalize mistakes on rare classes more.

```python
# Class 0 appears 9x more than class 1 → give class 1 a higher weight
class_weights = torch.tensor([1.0, 9.0])

criterion = nn.CrossEntropyLoss(weight=class_weights)

predictions = torch.tensor([[2.0, 1.0], [0.5, 2.5]])
targets     = torch.tensor([0, 1])

loss = criterion(predictions, targets)
# mistakes on class 1 now cost 9x more
```

---

## 7. Custom Loss Functions

You can write any differentiable function as a loss — PyTorch will handle the gradient automatically.

```python
# Example: weighted MSE — penalize under-predictions more than over-predictions
def asymmetric_mse(predictions, targets):
    diff = predictions - targets
    weight = torch.where(diff < 0, torch.tensor(2.0), torch.tensor(1.0))
    return (weight * diff ** 2).mean()

loss = asymmetric_mse(predictions, targets)
loss.backward()   # gradients work automatically
```

As a class (reusable, consistent with PyTorch style):

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, predictions, targets):
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)               # probability of correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

criterion = FocalLoss(gamma=2.0)
loss = criterion(predictions, targets)
```

---

## 8. Quick Reference

| Loss | Task | Input | Target |
|---|---|---|---|
| `nn.CrossEntropyLoss` | Multi-class classification | `(N, C)` logits | `(N,)` class indices |
| `nn.BCELoss` | Binary classification | `(N,)` probabilities | `(N,)` 0 or 1 |
| `nn.BCEWithLogitsLoss` | Binary classification | `(N,)` logits | `(N,)` 0 or 1 |
| `nn.NLLLoss` | Multi-class (low-level) | `(N, C)` log-probs | `(N,)` class indices |
| `nn.MSELoss` | Regression | `(N,)` or `(N, *)` | same shape |
| `nn.L1Loss` | Regression (robust) | `(N,)` or `(N, *)` | same shape |
| `nn.SmoothL1Loss` | Regression / detection | `(N,)` or `(N, *)` | same shape |

---

### Formulas

#### CrossEntropyLoss
```
L = -log( exp(x[y]) / Σ exp(x[j]) )
  = -x[y] + log( Σ exp(x[j]) )

x    : logits vector for one sample  (e.g. [2.0, 1.0, 0.5])
y    : correct class index           (e.g. 0)
x[y] : logit of the correct class
Σ    : sum over all classes
```

#### BCELoss
```
L = -[ y·log(p) + (1-y)·log(1-p) ]

p : predicted probability (after sigmoid), range (0, 1)
y : true label, 0 or 1

y=1 → L = -log(p)      high p → small loss, low p → large loss
y=0 → L = -log(1-p)    low p  → small loss, high p → large loss
```

#### BCEWithLogitsLoss
```
L = -[ y·log(σ(x)) + (1-y)·log(1-σ(x)) ]

x    : raw logit (before sigmoid)
σ(x) : sigmoid(x) = 1 / (1 + exp(-x))

Same as BCELoss but sigmoid is applied internally
```

#### NLLLoss
```
L = -log_prob[y]

log_prob : output of log_softmax  (e.g. [-0.46, -1.46, -1.96])
y        : correct class index
→ just picks the log-probability at the correct class and negates it
```

#### MSELoss
```
L = (1/N) · Σ (ŷᵢ - yᵢ)²

ŷ : predicted value
y : true value
→ average of squared differences
```

#### L1Loss
```
L = (1/N) · Σ |ŷᵢ - yᵢ|

→ average of absolute differences
→ less sensitive to outliers than MSE (no squaring)
```

#### SmoothL1Loss (Huber)
```
       | 0.5 · (ŷ - y)²        if |ŷ - y| < β   ← MSE region (smooth near 0)
Lᵢ =  |
       | β · (|ŷ - y| - 0.5β)  otherwise         ← L1 region (linear for large errors)

β = 1.0 by default
→ squared for small errors (stable gradient near 0)
→ linear for large errors  (not dominated by outliers)
```

---

**Key rules:**
- `CrossEntropyLoss` → raw logits, **no Softmax beforehand**
- `BCELoss` → apply Sigmoid yourself first
- `BCEWithLogitsLoss` → raw logits, Sigmoid applied internally ✅ preferred
- All losses return a scalar by default (`reduction='mean'`)

