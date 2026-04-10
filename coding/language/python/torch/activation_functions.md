# Activation Functions in PyTorch

Activation functions introduce **non-linearity** into neural networks. Without them, stacking linear layers is still just a linear transformation — the network could never learn complex patterns.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

## Table of Contents

1. [ReLU](#1-relu)
2. [LeakyReLU](#2-leakyrelu)
3. [ELU](#3-elu)
4. [Sigmoid](#4-sigmoid)
5. [Tanh](#5-tanh)
6. [Softmax](#6-softmax)
7. [GELU](#7-gelu)
8. [Swish / SiLU](#8-swish--silu)
9. [How to Use in a Model](#9-how-to-use-in-a-model)
10. [Quick Reference](#10-quick-reference)

---

## 1. ReLU

**Rectified Linear Unit** — the most widely used activation function.

**Formula:** `f(x) = max(0, x)`

```python
relu = nn.ReLU()

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
relu(x)
# tensor([0., 0., 0., 1., 3.])
```

```
output
  |       /
  |      /
  |     /
  |----/-------> input
  0
```

- Negatives → 0, positives → unchanged
- Fast to compute, works well in most cases
- **Problem: Dying ReLU** — neurons that output 0 for all inputs stop learning entirely (gradient = 0)

---

## 2. LeakyReLU

Fixes the dying ReLU problem by allowing a small slope for negatives.

**Formula:** `f(x) = x if x > 0, else α·x` (α is small, default 0.01)

```python
leaky = nn.LeakyReLU(negative_slope=0.01)

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
leaky(x)
# tensor([-0.03, -0.01,  0.00,  1.00,  3.00])
```

```
output
  |       /
  |      /
  |     /
--/----/-------> input
 /    0
/  (small slope)
```

- Negatives get a small gradient instead of 0 → neurons keep learning
- `negative_slope` is a hyperparameter you can tune

---

## 3. ELU

**Exponential Linear Unit** — smooth version of LeakyReLU for negatives.

**Formula:** `f(x) = x if x > 0, else α·(exp(x) - 1)`

```python
elu = nn.ELU(alpha=1.0)

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
elu(x)
# tensor([-0.9502, -0.6321,  0.0000,  1.0000,  3.0000])
```

- Smooth curve for negatives (no sharp corner at 0)
- Output for negatives approaches `-α` as x → -∞ (bounded below)
- Can produce negative outputs → mean activation closer to zero → faster learning

---

## 4. Sigmoid

Squashes any value into the range **(0, 1)**. Classic output for binary classification.

**Formula:** `f(x) = 1 / (1 + exp(-x))`

```python
sigmoid = nn.Sigmoid()

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
sigmoid(x)
# tensor([0.0474, 0.2689, 0.5000, 0.7311, 0.9526])
```

```
output
 1.0 |         ___---
 0.5 |      --/
 0.0 |---__/
     +-------------> input
```

- Output range: (0, 1) → interpret as **probability**
- **Problem: Vanishing gradient** — for very large or small inputs, gradient ≈ 0 → slow learning
- Mainly used in the **output layer** for binary classification, not hidden layers

---

## 5. Tanh

Like Sigmoid but output range is **(-1, 1)**. Zero-centered.

**Formula:** `f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`

```python
tanh = nn.Tanh()

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
tanh(x)
# tensor([-0.9951, -0.7616,  0.0000,  0.7616,  0.9951])
```

- Output range: (-1, 1) → **zero-centered** (better than Sigmoid for hidden layers)
- Still has vanishing gradient problem at extremes
- Common in RNNs / LSTMs for hidden states

---

## 6. Softmax

Converts a vector of raw scores (logits) into a **probability distribution** that sums to 1.

**Formula:** `f(xᵢ) = exp(xᵢ) / Σ exp(xⱼ)`

```python
softmax = nn.Softmax(dim=1)

x = torch.tensor([[2.0, 1.0, 0.5]])   # raw logits
softmax(x)
# tensor([[0.6241, 0.2297, 0.1462]])   ← sums to 1.0
```

- Used in the **output layer** for multi-class classification
- `dim` specifies which axis to normalize over (usually `dim=1` for batched inputs)
- **Note:** `nn.CrossEntropyLoss` applies softmax internally — do **not** add Softmax before it

```python
# Wrong — double softmax
model = nn.Sequential(..., nn.Softmax(dim=1))
loss = nn.CrossEntropyLoss()(output, target)   # already applies softmax internally

# Correct
model = nn.Sequential(...)                      # raw logits as output
loss = nn.CrossEntropyLoss()(output, target)
```

---

## 7. GELU

**Gaussian Error Linear Unit** — standard activation in modern Transformers (BERT, GPT).

**Formula:** `f(x) = x · Φ(x)` where Φ is the Gaussian CDF

```python
gelu = nn.GELU()

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
gelu(x)
# tensor([-0.0040, -0.1587,  0.0000,  0.8413,  2.9960])
```

- Smooth everywhere, including near 0 (unlike ReLU's sharp corner)
- Slightly penalizes small negative values instead of hard-zeroing them
- Default activation in BERT, GPT-2, ViT, and most modern Transformers

---

## 8. Swish / SiLU

**Swish** (also called **SiLU** — Sigmoid Linear Unit) — used in EfficientNet, newer models.

**Formula:** `f(x) = x · sigmoid(x)`

```python
silu = nn.SiLU()   # SiLU == Swish

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
silu(x)
# tensor([-0.1423, -0.2689,  0.0000,  0.7311,  2.8577])
```

- Smooth and non-monotonic (can decrease then increase)
- Empirically outperforms ReLU on deeper networks
- Slightly more expensive to compute than ReLU

---

## 9. How to Use in a Model

### As a layer inside `nn.Sequential`

```python
model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),           # ← between layers
    nn.Linear(64, 32),
    nn.GELU(),
    nn.Linear(32, 10),
    # no activation here — raw logits for CrossEntropyLoss
)
```

### Inside `forward()` using `F`

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))   # functional API — no need to define as attribute
        x = self.fc2(x)
        return x
```

### `nn.X()` vs `F.x()` — which to use?

| | `nn.ReLU()` (layer) | `F.relu()` (function) |
|---|---|---|
| Has state | No | No |
| Use in `nn.Sequential` | Yes | No |
| Use in `forward()` | Yes | Yes |
| Preference | For `Sequential` | For `forward()` |

Since activations have no learnable parameters, both are equivalent in behavior.

---

## 10. Quick Reference

| Activation | Output range | Formula | Best used for |
|---|---|---|---|
| `nn.ReLU` | [0, ∞) | max(0, x) | Default for hidden layers |
| `nn.LeakyReLU` | (-∞, ∞) | x or αx | When dying ReLU is a problem |
| `nn.ELU` | (-α, ∞) | x or α(eˣ-1) | Smooth alternative to LeakyReLU |
| `nn.Sigmoid` | (0, 1) | 1/(1+e⁻ˣ) | Binary classification output |
| `nn.Tanh` | (-1, 1) | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | RNNs, hidden states |
| `nn.Softmax` | (0, 1), sums to 1 | eˣⁱ/Σeˣʲ | Multi-class output |
| `nn.GELU` | (-0.17, ∞) approx | x·Φ(x) | Transformers (BERT, GPT) |
| `nn.SiLU` | (-0.28, ∞) approx | x·sigmoid(x) | EfficientNet, modern CNNs |
