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

## 5. BatchNorm

Normalizes activations across the batch to stabilize training.

```python
# After Linear
bn1d = nn.BatchNorm1d(128)
x = torch.randn(32, 128)    # (batch=32, features=128)
out = bn1d(x)

# After Conv2d
bn2d = nn.BatchNorm2d(16)
x = torch.randn(8, 16, 28, 28)   # (batch, channels, H, W)
out = bn2d(x)
```

- Use `BatchNorm1d` after `Linear`
- Use `BatchNorm2d` after `Conv2d`
- Typically placed **before** the activation function

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

## 7. Embedding

Lookup table that maps integer indices to dense vectors. Used for NLP.

```python
embed = nn.Embedding(num_embeddings=1000, embedding_dim=64)
# vocabulary of 1000 words, each mapped to a 64-dim vector

tokens = torch.tensor([5, 42, 100])   # word indices
out = embed(tokens)
print(out.shape)  # torch.Size([3, 64])
```

---

## 8. LSTM

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
