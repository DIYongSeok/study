# Optimizers & Schedulers in PyTorch

An **optimizer** updates model weights to minimize the loss. A **scheduler** adjusts the learning rate during training.

```python
import torch
import torch.nn as nn
import torch.optim as optim
```

## Table of Contents

1. [How Optimization Works](#1-how-optimization-works)
2. [Gradient Descent Variants](#2-gradient-descent-variants)
3. [Optimizers](#3-optimizers)
   - 3.1 [SGD](#31-sgd)
   - 3.2 [Adam](#32-adam)
   - 3.3 [AdamW](#33-adamw)
   - 3.4 [RMSprop](#34-rmsprop)
4. [Optimizer Internals](#4-optimizer-internals)
5. [Learning Rate Schedulers](#5-learning-rate-schedulers)
   - 5.1 [StepLR](#51-steplr)
   - 5.2 [CosineAnnealingLR](#52-cosineannealinglr)
   - 5.3 [ReduceLROnPlateau](#53-reducelronplateau)
   - 5.4 [OneCycleLR](#54-onecyclelr)
6. [Putting It All Together](#6-putting-it-all-together)
7. [Quick Reference](#7-quick-reference)

---

## 1. How Optimization Works

The goal of training is to find weights `w` that minimize the loss `L`. The optimizer does this by repeatedly:

```
1. Compute loss:        L = criterion(model(x), y)
2. Compute gradients:   L.backward()          → fills w.grad for every weight
3. Update weights:      optimizer.step()      → w = w - lr * w.grad
4. Clear gradients:     optimizer.zero_grad() → reset w.grad to 0 for next step
```

The core update rule for the simplest optimizer (SGD):

```
w ← w - lr · ∇L(w)

w   : current weight
lr  : learning rate (step size)
∇L  : gradient of loss with respect to w
```

**Learning rate** controls how large each step is:

```
lr too large:  overshoots the minimum → loss diverges
lr too small:  tiny steps → training takes forever

loss
 |  *
 |    *                  lr too large → bounces over the minimum
 |      *   *   *
 |        *
 +----------------------> steps

loss
 |  *
 |   *
 |    *                  lr too small → crawls toward minimum
 |     *
 |      *
 +----------------------> steps
```

---

## 2. Gradient Descent Variants

Before picking an optimizer, understand the three variants of gradient descent:

| Variant | Batch size | Update frequency | Pros | Cons |
|---|---|---|---|---|
| **Batch GD** | Entire dataset | Once per epoch | Stable, accurate gradient | Slow, memory heavy |
| **Stochastic GD (SGD)** | 1 sample | Every sample | Fast updates | Very noisy |
| **Mini-batch GD** | 32–512 samples | Every mini-batch | Balance of both ✅ | Slight noise |

PyTorch's `DataLoader` with `batch_size=32` does **mini-batch GD** — the standard in practice.

---

## 3. Optimizers

### 3.1 SGD

**Stochastic Gradient Descent** — the simplest optimizer.

**Formula:**
```
v ← momentum · v - lr · ∇L      (velocity, only if momentum > 0)
w ← w + v
```

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,       # smooths updates using past gradients
    weight_decay=1e-4,  # L2 regularization — penalizes large weights
    nesterov=True       # look-ahead version of momentum (slightly better)
)
```

**Momentum** accumulates past gradients like a ball rolling downhill — builds speed in consistent directions, dampens oscillations:

```
Without momentum:   zig-zags slowly toward minimum
With momentum:      smooth arc, faster convergence
```

- Simple and well-understood
- Requires careful learning rate tuning
- Often best for fine-tuning pretrained models (stable, predictable)

---

### 3.2 Adam

**Adaptive Moment Estimation** — the most widely used optimizer.

Maintains a **per-parameter** learning rate by tracking two moving averages:

```
m ← β₁·m + (1-β₁)·∇L          (1st moment — mean of gradients, like momentum)
v ← β₂·v + (1-β₂)·∇L²         (2nd moment — mean of squared gradients)

m̂ = m / (1 - β₁ᵗ)             (bias correction — compensates for cold start)
v̂ = v / (1 - β₂ᵗ)

w ← w - lr · m̂ / (√v̂ + ε)
```

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,            # default — usually works well
    betas=(0.9, 0.999), # (β₁, β₂) — decay rates for moment estimates
    eps=1e-8,           # ε — prevents division by zero
    weight_decay=0      # L2 regularization (but see AdamW for a better approach)
)
```

**Why it adapts:** `v` tracks how large each parameter's gradients have been.
- Parameter with large gradients → large `v` → small effective lr → takes smaller steps
- Parameter with small gradients → small `v` → large effective lr → takes larger steps

```
Rarely updated weights (e.g. embeddings) → get larger steps  → catch up faster
Frequently updated weights               → get smaller steps → don't overshoot
```

- Works well out of the box with `lr=1e-3`
- Default choice for most tasks

---

### 3.3 AdamW

**Adam with decoupled weight decay** — the preferred optimizer for Transformers and modern architectures.

The problem with Adam + `weight_decay`:

```
Adam updates:   w ← w - lr · m̂/(√v̂ + ε)
L2 in Adam:     gradient = ∇L + λ·w   ← weight decay added INTO the gradient
                → the adaptive scaling also scales the decay → wrong
```

AdamW fixes this by applying weight decay **separately**, after the gradient update:

```
w ← w - lr · m̂/(√v̂ + ε)   ← gradient update (same as Adam)
w ← w - lr · λ · w          ← weight decay applied directly to weights
```

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01   # now correctly decoupled
)
```

- Default optimizer for BERT, GPT, ViT, and most Transformer training
- If using weight decay, always prefer AdamW over Adam

---

### 3.4 RMSprop

**Root Mean Square Propagation** — predecessor to Adam.

```
v ← α·v + (1-α)·∇L²     (running average of squared gradients)
w ← w - lr · ∇L / √(v + ε)
```

```python
optimizer = optim.RMSprop(
    model.parameters(),
    lr=1e-3,
    alpha=0.99,    # decay rate for squared gradient average
    eps=1e-8
)
```

- Handles non-stationary objectives well
- Common in RNNs and reinforcement learning
- Adam generally outperforms it — use Adam unless you have a specific reason

---

## 4. Optimizer Internals

### Accessing and modifying state

```python
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Inspect parameter groups
print(optimizer.param_groups)
# [{'params': [...], 'lr': 0.001, 'betas': (0.9, 0.999), ...}]

# Change learning rate mid-training
optimizer.param_groups[0]['lr'] = 1e-4

# Inspect internal state (Adam's moment estimates)
optimizer.state   # dict mapping each parameter tensor → its m and v
```

### Different learning rates per layer

```python
optimizer = optim.Adam([
    {'params': model.encoder.parameters(), 'lr': 1e-4},  # pretrained → small lr
    {'params': model.head.parameters(),    'lr': 1e-3},  # new layer  → large lr
])
```

### Saving and loading optimizer state

```python
# Save
torch.save(optimizer.state_dict(), 'optimizer.pth')

# Load
optimizer.load_state_dict(torch.load('optimizer.pth'))
# important when resuming training — restores moment estimates, not just lr
```

---

## 5. Learning Rate Schedulers

A scheduler automatically adjusts the learning rate during training. Call `scheduler.step()` after each epoch (or step).

### 5.1 StepLR

Multiplies the learning rate by `gamma` every `step_size` epochs.

```python
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# epoch 0–9:   lr = 0.1
# epoch 10–19: lr = 0.01   (0.1 × 0.1)
# epoch 20–29: lr = 0.001  (0.01 × 0.1)

for epoch in range(30):
    train(...)
    scheduler.step()
```

```
lr
0.1  |________
0.01 |        |________
0.001|                 |________
     +---10---+---10---+---10---> epoch
```

---

### 5.2 CosineAnnealingLR

Decreases lr following a cosine curve from `lr` to `eta_min` over `T_max` epochs.

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50,      # half-period of the cosine
    eta_min=1e-6   # minimum lr
)
```

```
lr
0.1  |*
     | *
     |  **
     |    ***
     |       *****
1e-6 |            *************
     +-----------------------> epoch
            T_max=50
```

- Smooth decay — avoids sudden drops
- Widely used in image classification training
- Can be combined with **warm restarts** (`CosineAnnealingWarmRestarts`) to escape local minima

---

### 5.3 ReduceLROnPlateau

Reduces lr when a monitored metric stops improving.

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',      # 'min' for loss, 'max' for accuracy
    factor=0.1,      # lr = lr × factor when triggered
    patience=5,      # wait 5 epochs before reducing
    min_lr=1e-6
)

for epoch in range(100):
    train_loss = train(...)
    val_loss = validate(...)
    scheduler.step(val_loss)   # pass the monitored metric
```

```
val_loss plateau detected after 5 epochs → lr reduced:
epoch 0–14:  lr = 0.01
epoch 15:    no improvement for 5 epochs → lr = 0.001
epoch 30:    no improvement for 5 epochs → lr = 0.0001
```

- Adaptive — reacts to actual training dynamics
- Good when you don't know in advance when to decay

---

### 5.4 OneCycleLR

Increases lr from a low value to `max_lr`, then decreases it back. Completes in one cycle.

```python
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    steps_per_epoch=len(train_loader),
    epochs=30
)

# Call after every BATCH, not every epoch
for epoch in range(30):
    for batch in train_loader:
        train_step(...)
        scheduler.step()   # ← per batch
```

```
lr
0.01 |       *
     |     *   *
     |   *       *
     | *           *
1e-4 |*              ************
     +------------------------------> steps
        warmup      cooldown
```

- Very effective — often achieves good results faster than other schedules
- Designed to be used with `max_lr` found via a **learning rate finder**

---

## 6. Putting It All Together

```python
model = CNN()
criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

for epoch in range(50):
    # --- Training ---
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()       # 1. clear old gradients
        output = model(x)           # 2. forward pass
        loss = criterion(output, y) # 3. compute loss
        loss.backward()             # 4. compute gradients
        optimizer.step()            # 5. update weights

    # --- Scheduler step (once per epoch) ---
    scheduler.step()

    print(f"Epoch {epoch+1}  lr={scheduler.get_last_lr()[0]:.6f}  loss={loss.item():.4f}")
```

### Gradient clipping (optional but common with RNNs/Transformers)

Large gradients can destabilize training. Clip them before `optimizer.step()`:

```python
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip here
optimizer.step()
```

---

## 7. Quick Reference

### Optimizers

| Optimizer | Formula (simplified) | Best for |
|---|---|---|
| `SGD` | `w -= lr · ∇L` | Fine-tuning, CV with careful tuning |
| `SGD + momentum` | `w -= lr · (β·v + ∇L)` | Image classification (e.g. ResNet) |
| `Adam` | `w -= lr · m̂ / (√v̂ + ε)` | Default for most tasks |
| `AdamW` | Adam + decoupled `λ·w` decay | Transformers, NLP, modern architectures |
| `RMSprop` | `w -= lr · ∇L / √(v + ε)` | RNNs, reinforcement learning |

### Schedulers

| Scheduler | Behavior | When to use |
|---|---|---|
| `StepLR` | Drops lr by `gamma` every N epochs | Simple, predictable decay |
| `CosineAnnealingLR` | Smooth cosine decay to `eta_min` | Image classification |
| `ReduceLROnPlateau` | Reduces lr when metric stagnates | When decay timing is unknown |
| `OneCycleLR` | Warm-up then decay in one cycle | Fast training, superconvergence |

### Key rules
- Always call `optimizer.zero_grad()` **before** `loss.backward()`
- Call `scheduler.step()` **after** `optimizer.step()`
- `ReduceLROnPlateau` needs a metric passed: `scheduler.step(val_loss)`
- `OneCycleLR` steps **per batch**, all others step **per epoch**
- Prefer `AdamW` over `Adam` when using weight decay
