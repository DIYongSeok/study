# Common Layers in PyTorch (`torch.nn`)

```python
import torch
import torch.nn as nn
```

## Table of Contents

1. [Linear](#1-linear)
2. [ReLU & Activation Functions](#2-relu--activation-functions)
3. [Conv2d](#3-conv2d)
4. [MaxPool2d](#4-maxpool2d)
5. [BatchNorm](#5-batchnorm)
6. [Dropout](#6-dropout)
7. [Embedding](#7-embedding)
8. [LSTM](#8-lstm)
9. [MultiheadAttention](#9-multiheadattention)
10. [Sequential](#10-sequential)
11. [Full CNN Example](#11-full-cnn-example)
12. [Quick Reference](#12-quick-reference)

---

## 1. Linear

Fully connected layer. Computes `y = xW^T + b`.

```python
layer = nn.Linear(4, 2)   # input size 4 → output size 2

x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # shape (1, 4)
out = layer(x)
print(out.shape)  # torch.Size([1, 2])
```

- Weight shape: `(out_features, in_features)`
- Bias shape: `(out_features,)`
- Most common layer in neural networks

---

## 2. ReLU & Activation Functions

`nn.ReLU` — `f(x) = max(0, x)`. Kills negative values.

```python
relu = nn.ReLU()

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
relu(x)
# tensor([0., 0., 0., 1., 2.])
```

Other activation variants:

```python
nn.LeakyReLU(negative_slope=0.01)  # small slope for negatives
nn.Sigmoid()                        # output range (0, 1)
nn.Tanh()                           # output range (-1, 1)
nn.GELU()                           # used in Transformers
```

---

## 3. Conv2d

2D convolution layer. Mainly used for image data.

```python
conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
# input: (batch, 1, H, W) → output: (batch, 16, H, W)  [padding keeps size]

x = torch.randn(8, 1, 28, 28)   # 8 grayscale images 28x28
out = conv(x)
print(out.shape)  # torch.Size([8, 16, 28, 28])
```

Key parameters:

| Parameter | Meaning |
|---|---|
| `in_channels` | Number of input channels (e.g. 3 for RGB) |
| `out_channels` | Number of filters / feature maps |
| `kernel_size` | Size of the sliding window (e.g. 3 = 3×3) |
| `stride` | Step size of the window (default 1) |
| `padding` | Zeros added around the border to control output size |

---

## 4. MaxPool2d

Downsamples by taking the max value in each window.

```python
pool = nn.MaxPool2d(kernel_size=2)   # halves H and W

x = torch.randn(8, 16, 28, 28)
out = pool(x)
print(out.shape)  # torch.Size([8, 16, 14, 14])
```

- `nn.AvgPool2d` — takes the average instead of max
- No learnable parameters

---

## 5. [BatchNorm](https://youtu.be/LpXeOt3xmhM)

### What is Batch Normalization?

During training, as weights update, the distribution of inputs to each layer keeps shifting. This forces every layer to constantly re-adapt to new input distributions — slowing down training. This problem is called **internal covariate shift**.

**Batch Normalization** fixes this by normalizing the activations within each mini-batch — forcing them to have mean ≈ 0 and variance ≈ 1 before passing to the next layer.

---

### The formula

For each mini-batch:

```
Step 1 — compute mean:    μ = mean(x)
Step 2 — compute variance: σ² = var(x)
Step 3 — normalize:        x̂ = (x - μ) / √(σ² + ε)    ← ε prevents division by zero
Step 4 — scale & shift:    y = γ · x̂ + β               ← γ, β are learned parameters
```

`γ` (gamma) and `β` (beta) are learnable — the model can "undo" normalization if needed.

```python
bn = nn.BatchNorm1d(4)
print(bn.weight)   # γ — initialized to 1
print(bn.bias)     # β — initialized to 0
```

---

### Step-by-step example

```python
bn = nn.BatchNorm1d(3)
bn.eval()   # fix running stats for demonstration

x = torch.tensor([[1.0, 2.0, 3.0],
                  [3.0, 4.0, 5.0],
                  [5.0, 6.0, 7.0]])
# shape (3, 3) — batch of 3 samples, 3 features each

# For feature 0: values are [1, 3, 5] → mean=3, std=2
# normalized: [(1-3)/2, (3-3)/2, (5-3)/2] = [-1, 0, 1]

out = bn(x)
# tensor([[-1.,  -1.,  -1.],
#         [ 0.,   0.,   0.],
#         [ 1.,   1.,   1.]])  ← each feature column normalized independently
```

---

### BatchNorm1d vs BatchNorm2d

The difference is **which dimensions get normalized over**.

#### BatchNorm1d — for Linear layers

Input shape: `(batch, features)`

Normalizes each **feature** across the batch dimension.

```python
bn1d = nn.BatchNorm1d(num_features=4)

x = torch.tensor([[1., 2., 3., 4.],   # sample 1
                  [5., 6., 7., 8.],   # sample 2
                  [9., 10., 11., 12.]])  # sample 3
# shape: (3, 4) — batch=3, features=4

# Feature 0: values = [1, 5, 9]  → normalize these 3 values
# Feature 1: values = [2, 6, 10] → normalize these 3 values
# ...one mean/variance computed per feature column

out = bn1d(x)
print(out.shape)  # torch.Size([3, 4])
```

```
Batch direction ↓    Feature →
 sample 1:  [ 1,  2,  3,  4 ]
 sample 2:  [ 5,  6,  7,  8 ]   ← normalize down each column
 sample 3:  [ 9, 10, 11, 12 ]
              ↑   ↑   ↑   ↑
           col0 col1 col2 col3   (4 means, 4 variances computed)
```

---

#### BatchNorm2d — for Conv2d layers

Input shape: `(batch, channels, H, W)`

Normalizes each **channel** across batch, height, and width.

```python
bn2d = nn.BatchNorm2d(num_features=3)   # 3 channels

x = torch.randn(8, 3, 28, 28)   # 8 images, 3 channels (RGB), 28×28
# shape: (8, 3, 28, 28)

# Channel 0 (Red):   collect all 8×28×28 = 6272 values → compute 1 mean, 1 variance
# Channel 1 (Green): collect all 8×28×28 = 6272 values → compute 1 mean, 1 variance
# Channel 2 (Blue):  collect all 8×28×28 = 6272 values → compute 1 mean, 1 variance

out = bn2d(x)
print(out.shape)  # torch.Size([8, 3, 28, 28])
```

```
For each channel, normalize across ALL of: batch × H × W
→ 3 channels = 3 means, 3 variances computed
```

---

### train() vs eval() mode

#### During training — batch statistics

Each mini-batch computes its own mean and variance on the fly:

```python
# batch of 3 samples, feature values: [1, 5, 9]
# mean  = (1 + 5 + 9) / 3 = 5.0
# var   = ((1-5)² + (5-5)² + (9-5)²) / 3 = 10.67
# normalized: [(1-5)/√10.67, (5-5)/√10.67, (9-5)/√10.67]
#           = [-1.22, 0.0, 1.22]
```

These batch statistics are **temporary** — used only for this step, then discarded.

---

#### During training — running averages are silently updated

At every training step, BatchNorm also quietly updates two stored values:
`running_mean` and `running_var`, using **exponential moving average (EMA)**:

```python
momentum = 0.1   # default in PyTorch

running_mean = (1 - momentum) * running_mean + momentum * batch_mean
running_var  = (1 - momentum) * running_var  + momentum * batch_var
```

This means each new batch contributes 10%, and the history keeps 90%. Over many batches, `running_mean` and `running_var` slowly converge to represent the **whole dataset**.

```python
bn = nn.BatchNorm1d(1)
print(bn.running_mean)   # tensor([0.])  ← starts at 0
print(bn.running_var)    # tensor([1.])  ← starts at 1

# after one training step with batch_mean=5.0:
# running_mean = 0.9 * 0.0 + 0.1 * 5.0 = 0.5

# after many steps the running_mean approaches the true dataset mean
```

You can inspect these at any time — they are **not** learnable parameters, just tracked statistics:

```python
bn = nn.BatchNorm1d(4)
print(bn.running_mean.shape)   # torch.Size([4]) — one per feature
print(bn.running_var.shape)    # torch.Size([4]) — one per feature
print(bn.weight.shape)         # torch.Size([4]) — γ, learnable
print(bn.bias.shape)           # torch.Size([4]) — β, learnable
```

---

#### During eval — running averages are used

At inference time you typically pass **one sample** (or a small batch), not a full batch. The batch mean and variance from a single sample are meaningless and noisy:

```python
# Single sample: [3.0]
# batch_mean = 3.0, batch_var = 0.0  ← useless, no population to measure from
```

So instead, `model.eval()` switches BatchNorm to use the stored `running_mean` and `running_var` — the stable estimates accumulated across the whole training set:

```python
model.eval()
# now BatchNorm uses: x̂ = (x - running_mean) / √(running_var + ε)
# running_mean and running_var are FIXED — not updated anymore
```

This gives **consistent, deterministic output** regardless of batch size.

```python
model.train()   # batch stats used  → running_mean/var updated
model.eval()    # running stats used → running_mean/var frozen
```

---

#### What goes wrong if you forget model.eval()

```python
# Wrong — model still in train() mode during inference
model.train()
with torch.no_grad():
    out = model(single_sample)
# BatchNorm computes mean/var from just 1 sample → batch_var = 0
# normalized value = (x - x) / √(0 + ε) = 0  ← all activations become 0!
```

This is why you must always call `model.eval()` before inference.

---

### Why it helps

| Without BatchNorm | With BatchNorm |
|---|---|
| Slow training — needs careful LR tuning | Faster training — tolerates higher learning rates |
| Sensitive to weight initialization | Less sensitive |
| Gradients can vanish/explode in deep nets | Stable gradient flow |
| Internal covariate shift | Distributions stay normalized |

- Use `BatchNorm1d` after `Linear`
- Use `BatchNorm2d` after `Conv2d`
- Place it **between** the linear/conv layer and the activation function

```python
# Correct order
nn.Linear(128, 64),
nn.BatchNorm1d(64),   # normalize first
nn.ReLU(),            # then activate
```

---

## 6. Dropout

Randomly zeroes out elements during training to prevent overfitting.

```python
dropout = nn.Dropout(p=0.5)   # 50% chance of zeroing each element

x = torch.ones(1, 10)
out = dropout(x)
# tensor([[2., 0., 2., 0., 2., 2., 0., 0., 2., 2.]])  ← scaled up to keep expectation
```

- Only active during **training** — automatically disabled in `model.eval()` mode
- `nn.Dropout2d` — drops entire channels (for Conv layers)

---

## 7. [Embedding](https://wikidocs.net/64779)

Lookup table that maps integer indices to dense vectors. Used for NLP.

```python
embed = nn.Embedding(num_embeddings=1000, embedding_dim=64)
# vocabulary of 1000 words, each mapped to a 64-dim vector

tokens = torch.tensor([5, 42, 100])   # word indices
out = embed(tokens)
print(out.shape)  # torch.Size([3, 64])
```

---

## 8. [LSTM](https://wikidocs.net/60762)

Long Short-Term Memory — processes sequential data.

```python
lstm = nn.LSTM(input_size=10, hidden_size=32, num_layers=2, batch_first=True)

x = torch.randn(8, 20, 10)   # (batch=8, seq_len=20, input_size=10)
out, (h_n, c_n) = lstm(x)
print(out.shape)   # torch.Size([8, 20, 32])  ← output at each timestep
print(h_n.shape)   # torch.Size([2, 8, 32])   ← final hidden state (num_layers, batch, hidden)
```

- `batch_first=True` — input/output shape is `(batch, seq, feature)` instead of `(seq, batch, feature)`
- `h_n` — hidden state, `c_n` — cell state

---

## 9. MultiheadAttention

Core building block of Transformers.

```python
attn = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)

x = torch.randn(8, 20, 64)   # (batch=8, seq_len=20, embed_dim=64)
out, weights = attn(x, x, x) # query, key, value
print(out.shape)              # torch.Size([8, 20, 64])
```

---

## 10. Sequential

Chains layers together in order. Simplifies model definition.

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 10),
)

x = torch.randn(32, 784)
out = model(x)
print(out.shape)  # torch.Size([32, 10])
```

---

## 11. Full CNN Example

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B, 32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (B, 32, 14, 14)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (B, 64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (B, 64, 7, 7)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # (B, 64*7*7)
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CNN()
x = torch.randn(8, 1, 28, 28)
print(model(x).shape)  # torch.Size([8, 10])
```

---

## 12. Quick Reference

| Layer | Use case | Learnable params |
|---|---|---|
| `nn.Linear` | Fully connected, MLP | Yes |
| `nn.Conv2d` | Image features | Yes |
| `nn.MaxPool2d` | Downsample images | No |
| `nn.BatchNorm1d/2d` | Stabilize training | Yes (scale/shift) |
| `nn.Dropout` | Prevent overfitting | No |
| `nn.ReLU` | Activation | No |
| `nn.Embedding` | Word/token lookup | Yes |
| `nn.LSTM` | Sequences, NLP | Yes |
| `nn.MultiheadAttention` | Transformers | Yes |
| `nn.Sequential` | Stack layers cleanly | — |
