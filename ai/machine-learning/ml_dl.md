# Machine Learning & Deep Learning

A unified reference for the theory and practice behind modern ML/DL systems — from linear regression to Transformers.

## Table of Contents

1. [ML Fundamentals](#1-ml-fundamentals)
   - 1.1 [What is Machine Learning?](#11-what-is-machine-learning)
   - 1.2 [Learning Paradigms](#12-learning-paradigms)
   - 1.3 [Generalization & the Test Error Decomposition](#13-generalization--the-test-error-decomposition)
   - 1.4 [Data Splits & Cross-Validation](#14-data-splits--cross-validation)
   - 1.5 [Evaluation Metrics](#15-evaluation-metrics)
2. [Linear Models](#2-linear-models)
   - 2.1 [Linear Regression](#21-linear-regression)
   - 2.2 [Logistic Regression](#22-logistic-regression)
   - 2.3 [Support Vector Machine (SVM)](#23-support-vector-machine-svm)
3. [Tree-Based Models](#3-tree-based-models)
   - 3.1 [Decision Tree](#31-decision-tree)
   - 3.2 [Random Forest](#32-random-forest)
   - 3.3 [Gradient Boosting (XGBoost)](#33-gradient-boosting-xgboost)
4. [Unsupervised Learning](#4-unsupervised-learning)
   - 4.1 [K-Means Clustering](#41-k-means-clustering)
   - 4.2 [Gaussian Mixture Model (GMM)](#42-gaussian-mixture-model-gmm)
   - 4.3 [PCA & Dimensionality Reduction](#43-pca--dimensionality-reduction)
5. [Neural Networks](#5-neural-networks)
   - 5.1 [The Perceptron & MLP](#51-the-perceptron--mlp)
   - 5.2 [Activation Functions](#52-activation-functions)
   - 5.3 [Forward Pass](#53-forward-pass)
6. [Backpropagation](#6-backpropagation)
   - 6.1 [The Chain Rule in Computation Graphs](#61-the-chain-rule-in-computation-graphs)
   - 6.2 [Worked Example: Linear + MSE](#62-worked-example-linear--mse)
   - 6.3 [Vanishing & Exploding Gradients](#63-vanishing--exploding-gradients)
7. [Optimization](#7-optimization)
   - 7.1 [Gradient Descent Variants](#71-gradient-descent-variants)
   - 7.2 [Momentum & Nesterov](#72-momentum--nesterov)
   - 7.3 [Adam & Adaptive Methods](#73-adam--adaptive-methods)
   - 7.4 [Learning Rate Scheduling](#74-learning-rate-scheduling)
   - 7.5 [Gradient Clipping](#75-gradient-clipping)
8. [Regularization](#8-regularization)
   - 8.1 [L1 & L2 Regularization](#81-l1--l2-regularization)
   - 8.2 [Dropout](#82-dropout)
   - 8.3 [Batch Normalization](#83-batch-normalization)
   - 8.4 [Layer Normalization](#84-layer-normalization)
   - 8.5 [Early Stopping & Data Augmentation](#85-early-stopping--data-augmentation)
9. [Convolutional Neural Networks (CNN)](#9-convolutional-neural-networks-cnn)
   - 9.1 [Convolution Operation](#91-convolution-operation)
   - 9.2 [Pooling & Receptive Field](#92-pooling--receptive-field)
   - 9.3 [Classic Architectures](#93-classic-architectures)
10. [Recurrent Neural Networks (RNN)](#10-recurrent-neural-networks-rnn)
    - 10.1 [Vanilla RNN](#101-vanilla-rnn)
    - 10.2 [LSTM](#102-lstm)
    - 10.3 [GRU](#103-gru)
11. [Attention & Transformers](#11-attention--transformers)
    - 11.1 [Scaled Dot-Product Attention](#111-scaled-dot-product-attention)
    - 11.2 [Multi-Head Attention](#112-multi-head-attention)
    - 11.3 [Positional Encoding](#113-positional-encoding)
    - 11.4 [Transformer Architecture](#114-transformer-architecture)
    - 11.5 [BERT, GPT, and Variants](#115-bert-gpt-and-variants)
12. [Loss Functions](#12-loss-functions)
    - 12.1 [Regression Losses](#121-regression-losses)
    - 12.2 [Classification Losses](#122-classification-losses)
    - 12.3 [Contrastive & Metric Learning Losses](#123-contrastive--metric-learning-losses)
13. [Training Techniques](#13-training-techniques)
    - 13.1 [Weight Initialization](#131-weight-initialization)
    - 13.2 [Batch Size & Training Dynamics](#132-batch-size--training-dynamics)
    - 13.3 [Mixed Precision Training](#133-mixed-precision-training)
    - 13.4 [Transfer Learning & Fine-Tuning](#134-transfer-learning--fine-tuning)

---

## 1. ML Fundamentals

### 1.1 What is Machine Learning?

Machine Learning is the study of algorithms that **learn from data** to make predictions or decisions, without being explicitly programmed for each case.

$$f_\theta : \mathcal{X} \to \mathcal{Y}$$

| Symbol | Meaning |
|--------|---------|
| $\mathcal{X}$ | Input space (features) |
| $\mathcal{Y}$ | Output space (labels or predictions) |
| $\theta$ | Learnable parameters |
| $f_\theta$ | Model — maps inputs to outputs |

**The ML workflow:**

```
Data → Feature Engineering → Model → Training (minimize loss) → Evaluation → Deployment
```

**Core assumption:** the training data and test data are drawn from the same distribution — i.i.d. (independent and identically distributed).

### 1.2 Learning Paradigms

| Paradigm | Training Signal | Examples |
|----------|----------------|---------|
| **Supervised** | Labeled pairs $(x, y)$ | Classification, Regression |
| **Unsupervised** | Only $x$, no labels | Clustering, PCA, Generative models |
| **Self-supervised** | Labels derived from data itself | Masked language modeling (BERT), contrastive (SimCLR) |
| **Reinforcement** | Reward signal from environment | Game playing, robotics |
| **Semi-supervised** | Few labeled + many unlabeled | Label propagation, pseudo-labeling |

**Supervised learning objective:**

$$\hat{\theta} = \arg\min_\theta \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f_\theta(x_i),\, y_i)$$

| Term | Meaning |
|------|---------|
| $\mathcal{L}$ | Loss function — measures prediction error |
| $f_\theta(x_i)$ | Model prediction for input $x_i$ |
| $y_i$ | True label |
| $N$ | Number of training samples |

### 1.3 Generalization & the Test Error Decomposition

A model's **expected test error** decomposes as:

$$\mathbb{E}[\text{Test Error}] = \text{Bias}^2 + \text{Variance} + \sigma^2_\epsilon$$

| Term | Cause | Symptom |
|------|-------|---------|
| **Bias²** | Model too simple | Train error high, val error high |
| **Variance** | Model too complex | Train error low, val error high |
| **Noise $\sigma^2_\epsilon$** | Irreducible data noise | Cannot fix |

**Underfitting vs Overfitting:**

```
Train Loss  ↑ high    low     low
Val Loss    ↑ high    low     high
State       │ Under   Good    Overfit
```

The **inductive bias** of a model is its built-in assumption about the problem:

| Model | Inductive Bias |
|-------|---------------|
| Linear regression | Relationship is linear |
| CNN | Spatial locality, translation invariance |
| RNN | Sequential dependencies |
| Transformer | Pairwise token interactions (attention) |

### 1.4 Data Splits & Cross-Validation

**Train / Validation / Test split:**

```
Dataset
├── Train set   (60-80%) — used for gradient updates
├── Val set     (10-20%) — used for hyperparameter tuning
└── Test set    (10-20%) — used for final evaluation only
```

> Never tune hyperparameters on the test set — this leads to optimistic (overfit) estimates of generalization.

**K-Fold Cross-Validation:**

Split data into $K$ folds; train on $K-1$, validate on the remaining 1, rotate $K$ times:

$$\text{CV Error} = \frac{1}{K}\sum_{k=1}^K \text{Val Error on fold } k$$

- $K = 5$ or $K = 10$ are common
- **Leave-One-Out (LOO):** $K = N$ — expensive but low bias

```python
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

### 1.5 Evaluation Metrics

**Regression:**

| Metric | Formula | Sensitivity |
|--------|---------|-------------|
| MAE | $$\frac{1}{N}\sum_{i=1}^{N}\lvert y_i - \hat{y}_i\rvert$$ | Robust to outliers |
| MSE | $\frac{1}{N}\sum(y_i - \hat{y}_i)^2$ | Penalizes large errors heavily |
| RMSE | $\sqrt{\text{MSE}}$ | Same units as $y$ |
| R² | $1 - \text{SSE}/\text{SST}$ | Fraction of variance explained |

**Classification (from confusion matrix):**

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}, \qquad \text{Recall} = \frac{TP}{TP + FN}$$

$$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\,TP}{2\,TP + FP + FN}$$

| Metric | Use when |
|--------|---------|
| Accuracy | Balanced classes |
| Precision | False positives are costly (spam detection) |
| Recall | False negatives are costly (cancer screening) |
| F1 | Imbalanced classes |
| AUC-ROC | Ranking quality; threshold-independent |

```python
from sklearn.metrics import classification_report, roc_auc_score

print(classification_report(y_true, y_pred))
auc = roc_auc_score(y_true, y_score)
```

---

## 2. Linear Models

### 2.1 Linear Regression

Models the output as a linear combination of input features:

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b = \sum_{j=1}^d w_j x_j + b$$

| Term | Meaning |
|------|---------|
| $\mathbf{w} \in \mathbb{R}^d$ | Weight vector — importance of each feature |
| $b \in \mathbb{R}$ | Bias (intercept) |
| $\hat{y}$ | Predicted output |

**Matrix form for $N$ samples:** absorb the bias into $\mathbf{w}$ by appending a column of ones to $X$, giving $X \in \mathbb{R}^{N \times (d+1)}$:

$$\hat{\mathbf{y}} = X\mathbf{w}$$

---

**Loss — Mean Squared Error:**

$$\mathcal{L}(\mathbf{w}) = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2 = \frac{1}{N}\lVert \mathbf{y} - X\mathbf{w} \rVert^2$$

---

**Closed-form solution (Normal Equation):**

$$\hat{\mathbf{w}} = (X^\top X)^{-1} X^\top \mathbf{y}$$

Works when $d$ is small and $X^\top X$ is invertible. For large $d$, use gradient descent instead (inverting an $d \times d$ matrix costs $O(d^3)$).

**Why does this give the minimum?** Three steps:

*Step 1 — Take the gradient and set it to zero.*

Expand the loss:
$$\mathcal{L} = \frac{1}{N}\bigl(\mathbf{y}^\top\mathbf{y} - 2\mathbf{w}^\top X^\top \mathbf{y} + \mathbf{w}^\top X^\top X\mathbf{w}\bigr)$$

Differentiate with respect to $\mathbf{w}$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{2}{N}\bigl(X^\top X\mathbf{w} - X^\top \mathbf{y}\bigr) = \mathbf{0}$$

This gives the **Normal Equations**:

$$X^\top X\,\hat{\mathbf{w}} = X^\top \mathbf{y}$$

*Step 2 — Solve.* If $X^\top X$ is invertible (columns of $X$ are linearly independent — no duplicate or perfectly correlated features):

$$\hat{\mathbf{w}} = (X^\top X)^{-1} X^\top \mathbf{y}$$

*Step 3 — Confirm it is a minimum, not a saddle point.*

Compute the Hessian of $\mathcal{L}$:

$$H = \frac{\partial^2 \mathcal{L}}{\partial \mathbf{w}^2} = \frac{2}{N} X^\top X$$

For any non-zero vector $\mathbf{v}$:

$$\mathbf{v}^\top (X^\top X)\,\mathbf{v} = \lVert X\mathbf{v} \rVert^2 \geq 0$$

So $X^\top X$ is always **positive semi-definite**. When $X$ has full column rank, $\lVert X\mathbf{v} \rVert^2 > 0$ for all $\mathbf{v} \neq \mathbf{0}$, making $H$ **positive definite** — the loss surface is strictly convex (a bowl with a single bottom). Any critical point of a strictly convex function is the unique global minimum. Therefore $\hat{\mathbf{w}}$ is guaranteed to minimize $\mathcal{L}$.

**Geometric interpretation:** $X\hat{\mathbf{w}}$ is the **orthogonal projection** of $\mathbf{y}$ onto the column space of $X$. The residual $\mathbf{y} - X\hat{\mathbf{w}}$ is perpendicular to every column of $X$:

$$X^\top(\mathbf{y} - X\hat{\mathbf{w}}) = \mathbf{0}$$

That is exactly the Normal Equation — hence the name.

---

**Gradient (used by gradient descent when closed form is too costly):**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = -\frac{2}{N} X^\top (\mathbf{y} - X\mathbf{w})$$

Update rule: $\mathbf{w} \leftarrow \mathbf{w} - \eta \,\dfrac{\partial \mathcal{L}}{\partial \mathbf{w}}$

---

**Regularization**

Both variants add a penalty term to shrink weights and prevent overfitting.

**Ridge (L2):** $\mathcal{L} = \dfrac{1}{N}\lVert\mathbf{y} - X\mathbf{w}\rVert^2 + \lambda\lVert\mathbf{w}\rVert_2^2$

Closed form: $\hat{\mathbf{w}} = (X^\top X + \lambda I)^{-1} X^\top \mathbf{y}$

The $\lambda I$ shift makes the matrix always invertible — even when $X$ doesn't have full rank — and shrinks all weights uniformly toward zero.

**Lasso (L1):** $\mathcal{L} = \dfrac{1}{N}\lVert\mathbf{y} - X\mathbf{w}\rVert^2 + \lambda\lVert\mathbf{w}\rVert_1$

No closed form — the $\lvert w_j \rvert$ terms are non-differentiable at zero, so coordinate descent or subgradient methods are required. Produces **sparse** solutions: some weights become exactly zero, performing automatic feature selection.

| | Ridge (L2) | Lasso (L1) |
|---|---|---|
| Penalty | $\lambda\lVert\mathbf{w}\rVert_2^2$ | $\lambda\lVert\mathbf{w}\rVert_1$ |
| Effect on weights | Shrinks all toward zero | Zeros out some completely |
| Solution | Closed form | Iterative only |
| Best when | Many small useful features | Sparse signal, feature selection needed |
| Invertibility | Always invertible (even rank-deficient $X$) | N/A |

```python
import torch
import torch.nn as nn

# Linear regression as a single linear layer
model = nn.Linear(in_features=10, out_features=1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for x_batch, y_batch in dataloader:
    pred = model(x_batch)
    loss = criterion(pred.squeeze(), y_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 2.2 Logistic Regression

Binary classification: output is the probability of class 1.

$$\hat{p} = \sigma(\mathbf{w}^\top \mathbf{x} + b), \quad \sigma(z) = \frac{1}{1 + e^{-z}}$$

| Term | Meaning |
|------|---------|
| $\sigma$ | Sigmoid function — squashes $(-\infty, \infty)$ to $(0,1)$ |
| $\hat{p}$ | Predicted probability of the positive class |
| Decision boundary | $\hat{p} > 0.5 \iff \mathbf{w}^\top\mathbf{x} + b > 0$ |

**Loss — Binary Cross-Entropy:**

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \left[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right]$$

**Gradient:**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{1}{N} X^\top (\hat{\mathbf{p}} - \mathbf{y})$$

The gradient has the same form as linear regression — only the prediction step differs ($\sigma$ vs identity).

**Multiclass:** use softmax + cross-entropy instead of sigmoid + BCE.

$$\hat{\mathbf{p}} = \text{softmax}(W\mathbf{x} + \mathbf{b}), \quad \mathcal{L} = -\sum_k y_k \log \hat{p}_k$$

### 2.3 Support Vector Machine (SVM)

Finds the **maximum-margin** hyperplane separating two classes.

**Hard-margin SVM (linearly separable data):**

$$\min_{\mathbf{w}, b} \|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 \;\forall i$$

| Term | Meaning |
|------|---------|
| $\mathbf{w}^\top \mathbf{x} + b = 0$ | Decision hyperplane |
| $2/\|\mathbf{w}\|$ | Margin width — maximized by minimizing $\|\mathbf{w}\|$ |
| Support vectors | Training points closest to the boundary (on the margin lines) |

**Soft-margin SVM (with slack):**

$$\min_{\mathbf{w}, b, \xi} \|\mathbf{w}\|^2 + C\sum_i \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i,\; \xi_i \geq 0$$

| Term | Meaning |
|------|---------|
| $\xi_i \geq 0$ | Slack — allows misclassification |
| $C$ | Trade-off between margin width and misclassification penalty |
| Large $C$ | Small margin, few errors (risk overfit) |
| Small $C$ | Large margin, more errors (risk underfit) |

**Kernel trick:** replace $\mathbf{x}^\top \mathbf{x}'$ with $K(\mathbf{x}, \mathbf{x}') = \phi(\mathbf{x})^\top \phi(\mathbf{x}')$ — computes inner product in a high-dimensional feature space without explicit $\phi$.

| Kernel | Formula | Intuition |
|--------|---------|-----------|
| Linear | $\mathbf{x}^\top \mathbf{x}'$ | No transformation |
| RBF (Gaussian) | $\exp(-\gamma\|\mathbf{x}-\mathbf{x}'\|^2)$ | Similarity decreases with distance |
| Polynomial | $(\mathbf{x}^\top\mathbf{x}' + r)^d$ | Degree-$d$ polynomial features |

---

## 3. Tree-Based Models

### 3.1 Decision Tree

Recursively partitions the input space by asking binary questions.

**Split criterion — Information Gain (for classification):**

$$\text{IG}(S, A) = H(S) - \sum_{v \in A} \frac{|S_v|}{|S|} H(S_v)$$

| Term | Meaning |
|------|---------|
| $H(S) = -\sum_k p_k \log p_k$ | Entropy of node $S$ |
| $S_v$ | Subset of $S$ where feature $A = v$ |
| $\text{IG}$ | Reduction in entropy after splitting on $A$ |

**Gini impurity** (alternative, used in sklearn):

$$\text{Gini}(S) = 1 - \sum_k p_k^2$$

**Overfitting control:**

| Hyperparameter | Effect |
|----------------|--------|
| `max_depth` | Limit tree depth |
| `min_samples_leaf` | Minimum samples at leaf |
| `min_samples_split` | Minimum samples to split a node |
| `max_features` | Features to consider per split |

```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)
clf.fit(X_train, y_train)
```

### 3.2 Random Forest

**Ensemble of decision trees** using bagging (bootstrap aggregating).

**Algorithm:**
1. For $T$ trees:
   - Bootstrap sample (sample $N$ with replacement from training set)
   - Grow a full tree, but at each split **only consider a random subset of $\sqrt{d}$ features**
2. Aggregate: majority vote (classification) or average (regression)

| Component | Purpose |
|-----------|---------|
| Bagging | Reduces variance — each tree sees different data |
| Feature subsampling | Decorrelates trees — prevents all trees from using the same dominant features |
| Ensemble | Average reduces variance without increasing bias |

**Out-of-bag (OOB) error:** samples not included in each bootstrap can be used as a built-in validation set — no need for explicit cross-validation.

**Feature importance:**

$$\text{Importance}(j) = \sum_{\text{trees}} \sum_{\text{nodes split on } j} \Delta \text{impurity} \cdot \frac{N_\text{node}}{N}$$

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', n_jobs=-1)
rf.fit(X_train, y_train)

importances = rf.feature_importances_   # per-feature importance scores
```

### 3.3 Gradient Boosting (XGBoost)

**Boosting:** sequentially train weak learners, each correcting the errors of all previous ones.

**Algorithm:**
1. Initialize: $F_0(\mathbf{x}) = \text{constant}$
2. For $m = 1, \ldots, M$:
   - Compute **pseudo-residuals**: $r_i = -\frac{\partial \mathcal{L}(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}$
   - Fit a tree $h_m$ to the pseudo-residuals
   - Update: $F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta\, h_m(\mathbf{x})$

| Term | Meaning |
|------|---------|
| $r_i$ | Pseudo-residual — negative gradient of loss w.r.t. prediction |
| $\eta$ | Learning rate — shrinks each tree's contribution |
| $h_m$ | $m$-th weak learner (shallow decision tree) |
| $F_m$ | Current ensemble prediction |

**Intuition:** the pseudo-residual tells tree $m$ where the current ensemble is still wrong.

**XGBoost improvements over vanilla Gradient Boosting:**
- Second-order Taylor expansion of the loss (uses Hessian)
- L1 and L2 regularization on tree weights
- Column subsampling (like Random Forest)
- Efficient sparse-aware split finding

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20)
```

| Hyperparameter | Typical Range | Effect |
|----------------|---------------|--------|
| `n_estimators` | 100–1000 | More trees = more capacity |
| `learning_rate` | 0.01–0.3 | Smaller → need more trees |
| `max_depth` | 3–8 | Deeper → more complex |
| `subsample` | 0.5–1.0 | Row sampling per tree |
| `reg_lambda` | 0–10 | L2 regularization |

---

## 4. Unsupervised Learning

### 4.1 K-Means Clustering

Partitions $N$ data points into $K$ clusters by minimizing within-cluster variance.

**Objective:**

$$\min_{C_1,\ldots,C_K} \sum_{k=1}^K \sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$$

| Term | Meaning |
|------|---------|
| $C_k$ | Set of points assigned to cluster $k$ |
| $\boldsymbol{\mu}_k = \frac{1}{|C_k|}\sum_{\mathbf{x}\in C_k} \mathbf{x}$ | Centroid of cluster $k$ |

**Algorithm (Lloyd's):**
1. Initialize $K$ centroids randomly
2. **Assign:** each point to its nearest centroid
3. **Update:** recompute each centroid as the mean of assigned points
4. Repeat until convergence

**Weaknesses:**

| Issue | Fix |
|-------|-----|
| Sensitive to initialization | K-Means++ initialization |
| Assumes spherical clusters | GMM with full covariance |
| Must specify $K$ | Elbow method, silhouette score |
| Sensitive to outliers | K-Medoids |

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=5, init='k-means++', n_init=10, random_state=42)
labels = km.fit_predict(X)
centroids = km.cluster_centers_
```

### 4.2 Gaussian Mixture Model (GMM)

Models data as a mixture of $K$ Gaussian distributions.

$$p(\mathbf{x}) = \sum_{k=1}^K \pi_k\, \mathcal{N}(\mathbf{x};\, \boldsymbol{\mu}_k, \Sigma_k)$$

| Term | Meaning |
|------|---------|
| $\pi_k$ | Mixture weight — prior probability of cluster $k$, $\sum_k \pi_k = 1$ |
| $\boldsymbol{\mu}_k$ | Mean of cluster $k$ |
| $\Sigma_k$ | Covariance of cluster $k$ — encodes shape and orientation |

**Fitted with Expectation-Maximization (EM):**

- **E-step:** compute soft assignments $r_{ik} = P(\text{cluster }k \mid \mathbf{x}_i)$
- **M-step:** update $\pi_k, \boldsymbol{\mu}_k, \Sigma_k$ using weighted statistics

> GMM is a soft-assignment version of K-Means. K-Means is GMM with spherical, equal covariance and hard assignment.

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=5, covariance_type='full')
gmm.fit(X)

probs = gmm.predict_proba(X)   # soft assignments (N, K)
labels = gmm.predict(X)        # hard assignments
```

### 4.3 PCA & Dimensionality Reduction

**PCA (Principal Component Analysis)** finds the directions of maximum variance.

$$\text{Maximize} \quad \mathbf{w}^\top \Sigma \mathbf{w} \quad \text{s.t.} \quad \|\mathbf{w}\| = 1$$

The solution is the eigenvectors of the covariance matrix $\Sigma = \frac{1}{N} X^\top X$:

$$\Sigma = V \Lambda V^\top, \quad \text{principal components} = \text{columns of } V$$

**Steps:**
1. Center data: $X \leftarrow X - \bar{X}$
2. Compute covariance matrix $\Sigma$ (or use SVD of $X$ directly)
3. Take top $k$ eigenvectors $V_k \in \mathbb{R}^{d \times k}$
4. Project: $Z = X V_k \in \mathbb{R}^{N \times k}$

**Explained variance ratio:**

$$\text{EVR}_j = \frac{\lambda_j}{\sum_i \lambda_i}$$

Choose $k$ to retain e.g. 95% of total variance.

| Method | Preserves | Good for |
|--------|-----------|---------|
| PCA | Global linear structure | Preprocessing, visualization |
| t-SNE | Local neighborhood structure | 2D/3D visualization |
| UMAP | Local + global structure | Faster t-SNE alternative |
| Autoencoder | Nonlinear structure | Representation learning |

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=50, svd_solver='full')
X_reduced = pca.fit_transform(X)

print(pca.explained_variance_ratio_.cumsum()[:10])   # cumulative EVR
```

---

## 5. Neural Networks

### 5.1 The Perceptron & MLP

**Single neuron:**

$$z = \mathbf{w}^\top \mathbf{x} + b, \quad a = \sigma(z)$$

| Term | Meaning |
|------|---------|
| $\mathbf{x}$ | Input vector |
| $\mathbf{w}$ | Weights — learned parameters |
| $b$ | Bias |
| $z$ | Pre-activation (linear combination) |
| $a = \sigma(z)$ | Activation (output of this neuron) |

**Multilayer Perceptron (MLP)** — stack of linear layers with nonlinear activations:

$$\mathbf{h}^{(l)} = \sigma\!\left(W^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\right), \quad l = 1, \ldots, L$$

| Layer | Input | Output | Size |
|-------|-------|--------|------|
| Input | $\mathbf{x}$ | — | $d_\text{in}$ |
| Hidden $l$ | $\mathbf{h}^{(l-1)}$ | $\mathbf{h}^{(l)}$ | $d_l$ |
| Output | $\mathbf{h}^{(L-1)}$ | $\hat{y}$ | $d_\text{out}$ |

**Universal approximation theorem:** an MLP with a single hidden layer and a nonlinear activation can approximate any continuous function on a compact domain to arbitrary accuracy — given enough hidden units.

```python
import torch.nn as nn

mlp = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)
```

### 5.2 Activation Functions

| Activation | Formula | Output | Used in |
|-----------|---------|--------|---------|
| Sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}$ | $(0,1)$ | Binary output |
| Tanh | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1,1)$ | RNN hidden state |
| ReLU | $\max(0, z)$ | $[0, \infty)$ | Hidden layers (default) |
| Leaky ReLU | $\max(\alpha z, z)$, $\alpha \ll 1$ | $(-\infty, \infty)$ | Fix dying ReLU |
| GELU | $z \cdot \Phi(z)$ | $(-\infty, \infty)$ | Transformers (BERT, GPT) |
| SiLU (Swish) | $z \cdot \sigma(z)$ | $(-\infty, \infty)$ | Modern networks |
| Softmax | $\frac{e^{z_k}}{\sum_j e^{z_j}}$ | $(0,1)^K$, sums to 1 | Multiclass output |

**Dying ReLU problem:** if $z < 0$ always, the neuron outputs 0 and its gradient is also 0 — the neuron never updates. Leaky ReLU, ELU, or good initialization fixes this.

**Why nonlinearity matters:** without activation functions, stacking linear layers is equivalent to a single linear layer — no additional expressivity.

### 5.3 Forward Pass

A concrete example: 2-layer MLP for 3-class classification, batch size 1.

Input: $\mathbf{x} \in \mathbb{R}^4$, hidden: 3 units, output: 3 classes.

$$\mathbf{z}^{(1)} = W^{(1)}\mathbf{x} + \mathbf{b}^{(1)} \in \mathbb{R}^3$$

$$\mathbf{h}^{(1)} = \text{ReLU}(\mathbf{z}^{(1)})$$

$$\mathbf{z}^{(2)} = W^{(2)}\mathbf{h}^{(1)} + \mathbf{b}^{(2)} \in \mathbb{R}^3$$

$$\hat{\mathbf{p}} = \text{softmax}(\mathbf{z}^{(2)})$$

$$\mathcal{L} = -\log \hat{p}_{y}$$

**Parameter count:** $W^{(1)}: 3 \times 4 = 12$, $\mathbf{b}^{(1)}: 3$, $W^{(2)}: 3 \times 3 = 9$, $\mathbf{b}^{(2)}: 3$ → **27 total**.

```python
import torch
import torch.nn.functional as F

x = torch.randn(4)            # input
W1 = torch.randn(3, 4)
b1 = torch.zeros(3)
W2 = torch.randn(3, 3)
b2 = torch.zeros(3)

z1 = W1 @ x + b1              # pre-activation (3,)
h1 = F.relu(z1)               # activation (3,)
z2 = W2 @ h1 + b2             # logits (3,)
p  = F.softmax(z2, dim=0)     # probabilities (3,)

y = 1                         # true class
loss = -torch.log(p[y])       # cross-entropy loss
```

---

## 6. Backpropagation

### 6.1 The Chain Rule in Computation Graphs

Backpropagation computes $\frac{\partial \mathcal{L}}{\partial \theta}$ for all parameters $\theta$ using the **chain rule** on the computation graph.

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l)}} \cdot \frac{\partial \mathbf{z}^{(l)}}{\partial W^{(l)}}$$

The **error signal** (local gradient) at layer $l$:

$$\boldsymbol{\delta}^{(l)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l)}}$$

**Backprop recursion:**

$$\boldsymbol{\delta}^{(l)} = \left(W^{(l+1)}\right)^\top \boldsymbol{\delta}^{(l+1)} \odot \sigma'(\mathbf{z}^{(l)})$$

| Term | Meaning |
|------|---------|
| $\boldsymbol{\delta}^{(l)}$ | Error signal at layer $l$ |
| $(W^{(l+1)})^\top \boldsymbol{\delta}^{(l+1)}$ | Gradient backpropagated from the next layer |
| $\odot$ | Element-wise multiplication |
| $\sigma'(\mathbf{z}^{(l)})$ | Gradient of the activation function |

**Parameter gradients:**

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \boldsymbol{\delta}^{(l)} \left(\mathbf{h}^{(l-1)}\right)^\top, \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}$$

**Computational graph view:** forward pass builds the graph; backward pass traverses it in reverse, multiplying local gradients (Jacobians).

### 6.2 Worked Example: Linear + MSE

Model: $\hat{y} = \mathbf{w}^\top \mathbf{x}$, Loss: $\mathcal{L} = \frac{1}{2}(y - \hat{y})^2$.

**Forward:**

$$\hat{y} = w_1 x_1 + w_2 x_2 = 2 \cdot 3 + (-1) \cdot 4 = 2$$

$$\mathcal{L} = \frac{1}{2}(5 - 2)^2 = 4.5 \quad (y = 5)$$

**Backward:**

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = \hat{y} - y = 2 - 5 = -3$$

$$\frac{\partial \hat{y}}{\partial w_1} = x_1 = 3, \qquad \frac{\partial \hat{y}}{\partial w_2} = x_2 = 4$$

$$\frac{\partial \mathcal{L}}{\partial w_1} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w_1} = -3 \cdot 3 = -9$$

$$\frac{\partial \mathcal{L}}{\partial w_2} = -3 \cdot 4 = -12$$

**Update** (lr = 0.01):

$$w_1 \leftarrow 2 - 0.01 \cdot (-9) = 2.09, \quad w_2 \leftarrow -1 - 0.01 \cdot (-12) = -0.88$$

```python
import torch

x = torch.tensor([3.0, 4.0])
y = torch.tensor(5.0)
w = torch.tensor([2.0, -1.0], requires_grad=True)

y_hat = w @ x
loss  = 0.5 * (y - y_hat)**2

loss.backward()
print(w.grad)    # tensor([-9., -12.])
```

### 6.3 Vanishing & Exploding Gradients

**Vanishing gradients:** gradient $\to 0$ as it propagates through many layers.

For sigmoid: $\sigma'(z) = \sigma(z)(1-\sigma(z)) \leq 0.25$. After $L$ layers:

$$\|\boldsymbol{\delta}^{(1)}\| \approx \|\boldsymbol{\delta}^{(L)}\| \cdot (0.25)^L \to 0 \quad \text{exponentially}$$

**Exploding gradients:** gradient $\to \infty$.

If $\|W\| > 1$, repeated multiplication $W^L$ causes exponential growth.

| Problem | Symptom | Fix |
|---------|---------|-----|
| Vanishing | Gradients near zero, early layers don't train | ReLU, ResNets, LSTM, gradient clipping |
| Exploding | Loss spikes, NaN | Gradient clipping, smaller lr, BatchNorm |

**ResNet solution:** skip connections add a direct gradient path:

$$\mathbf{h}^{(l+1)} = \mathbf{h}^{(l)} + F(\mathbf{h}^{(l)})$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(l)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}^{(l+1)}} \cdot \left(I + \frac{\partial F}{\partial \mathbf{h}^{(l)}}\right)$$

The identity $I$ ensures gradients flow directly through the skip connection — no matter how small $\partial F/\partial \mathbf{h}^{(l)}$ is.

---

## 7. Optimization

### 7.1 Gradient Descent Variants

**Full-batch Gradient Descent:**

$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta; \mathcal{D})$$

Uses all $N$ samples per update — accurate but slow for large datasets.

**Stochastic Gradient Descent (SGD):**

$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta; x_i, y_i)$$

Single sample per update — fast but very noisy.

**Mini-batch SGD** (used in practice):

$$\theta \leftarrow \theta - \eta \cdot \frac{1}{B} \sum_{i \in \mathcal{B}} \nabla_\theta \mathcal{L}(\theta; x_i, y_i)$$

| Term | Meaning |
|------|---------|
| $\eta$ | Learning rate |
| $B$ | Batch size |
| $\mathcal{B}$ | Mini-batch of $B$ randomly sampled training examples |

| Variant | Batch size | Gradient quality | Speed |
|---------|-----------|-----------------|-------|
| Full-batch | $N$ | Exact | Slow |
| SGD | 1 | Very noisy | Fast |
| Mini-batch | 32–512 | Approx, stable | Practical |

### 7.2 Momentum & Nesterov

**Momentum** accumulates a velocity vector in gradient direction:

$$\mathbf{v} \leftarrow \beta \mathbf{v} + (1-\beta) \nabla_\theta \mathcal{L}$$

$$\theta \leftarrow \theta - \eta \mathbf{v}$$

| Term | Typical Value | Meaning |
|------|--------------|---------|
| $\beta$ | 0.9 | Momentum coefficient — how much of past velocity to retain |
| $\mathbf{v}$ | — | Velocity (exponential moving average of gradients) |

**Effect:** dampens oscillations in high-curvature directions; accelerates in low-curvature directions. Like a ball rolling downhill — picks up speed.

**Nesterov Accelerated Gradient (NAG):** evaluate gradient at the "lookahead" position:

$$\mathbf{v} \leftarrow \beta \mathbf{v} + \eta \nabla_\theta \mathcal{L}(\theta - \beta \mathbf{v})$$

$$\theta \leftarrow \theta - \mathbf{v}$$

Nesterov converges faster in theory and often in practice.

### 7.3 Adam & Adaptive Methods

**Adam (Adaptive Moment Estimation):** combines Momentum (first moment) and RMSProp (second moment).

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad \text{(first moment — mean)}$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \qquad \text{(second moment — variance)}$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \qquad \text{(bias correction)}$$

$$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

| Term | Typical Value | Meaning |
|------|--------------|---------|
| $\beta_1$ | 0.9 | Decay for mean estimate |
| $\beta_2$ | 0.999 | Decay for variance estimate |
| $\epsilon$ | $10^{-8}$ | Numerical stability |
| $\eta$ | $10^{-3}$ | Learning rate |

**Intuition:** parameters with consistently large gradients get smaller effective learning rates; parameters with small or inconsistent gradients get larger effective learning rates. Automatically adapts per-parameter learning rate.

**Comparison:**

| Optimizer | Adapts lr? | Momentum | Best for |
|-----------|-----------|----------|---------|
| SGD | No | No | Simple, well-tuned |
| SGD + Momentum | No | Yes | CV training |
| RMSProp | Yes | No | RNN |
| Adam | Yes | Yes | General (default) |
| AdamW | Yes | Yes | Transformers |

**AdamW** (weight decay decoupled from gradient update):

$$\theta_t = \theta_{t-1} - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right)$$

> Weight decay in Adam applies on top of the adaptive update — AdamW fixes this by applying weight decay directly to $\theta$, not to the gradient.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

### 7.4 Learning Rate Scheduling

A fixed learning rate is often suboptimal. Schedules adjust $\eta$ during training.

| Schedule | Formula / Behavior | Used for |
|----------|--------------------|---------|
| Step decay | Multiply by $\gamma$ every $k$ epochs | CV |
| Cosine annealing | $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1+\cos\frac{\pi t}{T})$ | General |
| Warmup + cosine | Linear warmup, then cosine decay | Transformers |
| Reduce on plateau | Halve $\eta$ when val loss stops improving | Adaptive |
| OneCycleLR | Fast increase then gradual decrease | Super-convergence |

**Warmup** is critical for Transformers: starting with a large learning rate on random weights causes instability. Linear warmup for first $N_\text{warmup}$ steps prevents this.

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Warmup then cosine decay
warmup = LinearLR(optimizer, start_factor=0.01, total_iters=1000)
cosine = CosineAnnealingLR(optimizer, T_max=9000)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[1000])
```

### 7.5 Gradient Clipping

Prevents exploding gradients by capping gradient norm:

$$\text{if} \quad \|\mathbf{g}\| > \tau: \quad \mathbf{g} \leftarrow \frac{\tau}{\|\mathbf{g}\|} \mathbf{g}$$

| Term | Meaning |
|------|---------|
| $\mathbf{g}$ | Gradient vector (all parameters concatenated) |
| $\tau$ | Clip threshold (typically 1.0) |
| $\|\mathbf{g}\|$ | Global gradient norm |

Essential for RNNs and Transformers, which are prone to gradient spikes.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

## 8. Regularization

### 8.1 L1 & L2 Regularization

Add a penalty on weights to the loss:

$$\mathcal{L}_\text{reg} = \mathcal{L} + \lambda R(\theta)$$

| Type | Penalty | Gradient | Effect |
|------|---------|---------|--------|
| **L2** (Ridge) | $\lambda \|\theta\|_2^2$ | $2\lambda\theta$ | Shrinks all weights toward zero (smooth) |
| **L1** (Lasso) | $\lambda \|\theta\|_1$ | $\lambda \text{sign}(\theta)$ | Pushes many weights to exactly zero (sparse) |

**L2 in practice:** called **weight decay** — implemented as an optimizer parameter:

```python
# Equivalent to adding λ‖θ‖² to the loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
```

**Why L1 gives sparsity:** L1's constant gradient $\lambda \text{sign}(\theta)$ pulls weights toward zero with the same force regardless of weight magnitude. Once a weight hits zero, it stays there. L2's gradient $2\lambda\theta$ shrinks with the weight, so it never quite reaches zero.

### 8.2 Dropout

During training, randomly zero out each neuron's output with probability $p$ (typically 0.1–0.5).

$$\tilde{\mathbf{h}} = \mathbf{h} \odot \mathbf{m}, \quad m_i \sim \text{Bernoulli}(1-p)$$

**At inference:** multiply outputs by $(1-p)$ to match expected value. In practice, PyTorch uses **inverted dropout** — divide by $(1-p)$ during training so no change at inference.

| Term | Meaning |
|------|---------|
| $p$ | Drop probability |
| $\mathbf{m}$ | Binary mask — 1 = keep, 0 = drop |
| Inverted dropout | Scale by $\frac{1}{1-p}$ during train → no scale needed at test |

**Why it works:**
- Prevents co-adaptation — neurons can't rely on specific others always being present
- Implicit ensemble of $2^d$ sub-networks
- Bayesian interpretation: approximate posterior sampling over network weights

```python
nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(p=0.3),    # 30% of neurons dropped during training
    nn.Linear(256, 10),
)
```

### 8.3 Batch Normalization

Normalizes each mini-batch's activations to have zero mean and unit variance, then applies learned scale and shift.

$$\text{BN}(\mathbf{z}) = \gamma \cdot \frac{\mathbf{z} - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}} + \beta$$

| Term | Meaning |
|------|---------|
| $\mu_\mathcal{B} = \frac{1}{B}\sum_i z_i$ | Mini-batch mean |
| $\sigma^2_\mathcal{B} = \frac{1}{B}\sum_i (z_i - \mu_\mathcal{B})^2$ | Mini-batch variance |
| $\gamma, \beta$ | Learnable scale and shift — restored expressivity |
| $\epsilon$ | Numerical stability (typically $10^{-5}$) |

**At inference:** use **running statistics** (exponential moving average from training) instead of batch statistics.

**Benefits:**
- Reduces **internal covariate shift** — distribution of layer inputs stays stable
- Allows higher learning rates
- Acts as regularization (noise from batch statistics)
- Makes deeper networks much easier to train

**Where to place BN:** typically after linear/conv, before activation — though debate exists (sometimes after activation works better).

```python
nn.Sequential(
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),   # BN after linear, before activation
    nn.ReLU(),
)
```

### 8.4 Layer Normalization

Normalizes across the **feature** dimension (not the batch dimension):

$$\text{LN}(\mathbf{z}) = \gamma \cdot \frac{\mathbf{z} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

$$\mu = \frac{1}{d}\sum_{j=1}^d z_j, \qquad \sigma^2 = \frac{1}{d}\sum_{j=1}^d (z_j - \mu)^2$$

| | Batch Norm | Layer Norm |
|--|-----------|-----------|
| Normalizes over | Batch (per feature) | Features (per sample) |
| Batch size dependent | Yes | No |
| Works for batch=1 | No | Yes |
| Used in | CNNs | Transformers, RNNs |

Layer Norm is **independent of batch size** — critical for autoregressive generation (batch size 1) and Transformers.

### 8.5 Early Stopping & Data Augmentation

**Early Stopping:** monitor validation loss; stop training when it starts to increase.

```python
class EarlyStopping:
    def __init__(self, patience=10, delta=1e-4):
        self.patience = patience; self.delta = delta
        self.best_loss = float('inf'); self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss; self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience   # True = stop
```

**Data Augmentation:** artificially expand the training set via label-preserving transformations.

| Domain | Common Augmentations |
|--------|---------------------|
| Images | Random crop, flip, rotation, color jitter, Cutout, MixUp |
| Text | Synonym replacement, back-translation, random deletion |
| Audio | Time stretch, pitch shift, SpecAugment (mask frequency/time bands) |
| Tabular | SMOTE (synthetic minority oversampling) |

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
```

---

## 9. Convolutional Neural Networks (CNN)

### 9.1 Convolution Operation

A learnable **filter** (kernel) slides over the input and computes dot products.

$$(\mathbf{I} * \mathbf{K})_{i,j} = \sum_{m}\sum_{n} \mathbf{I}_{i+m,\, j+n} \cdot \mathbf{K}_{m,n}$$

| Term | Meaning |
|------|---------|
| $\mathbf{I}$ | Input feature map (H × W × C) |
| $\mathbf{K}$ | Kernel/filter (k × k × C × C_out) |
| Stride $s$ | Step size of the sliding window |
| Padding $p$ | Zeros added around input border |

**Output size:**

$$H_\text{out} = \left\lfloor \frac{H_\text{in} + 2p - k}{s} \right\rfloor + 1$$

**Parameter count:** a conv layer with $C_\text{in}$ input channels, $C_\text{out}$ filters, kernel $k \times k$:

$$\text{Params} = k \times k \times C_\text{in} \times C_\text{out} + C_\text{out}\text{(bias)}$$

**Key properties:**
- **Parameter sharing:** same kernel applied everywhere → huge reduction in parameters vs fully-connected
- **Translation equivariance:** if the input shifts, the output shifts by the same amount
- **Local connectivity:** each output depends only on a small input region

**Depthwise Separable Convolution** (used in MobileNet):

- Depthwise: $k \times k$ per channel independently → $k^2 C_\text{in}$ params
- Pointwise: $1 \times 1$ convolution to mix channels → $C_\text{in} C_\text{out}$ params
- Total: $k^2 C_\text{in} + C_\text{in} C_\text{out}$ vs $k^2 C_\text{in} C_\text{out}$ — ~$8\times$ fewer for $k=3$

```python
import torch.nn as nn

# Standard conv: 32 filters, 3x3 kernel, padding=1 → same spatial size
conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)

# Depthwise separable
dw  = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3)   # depthwise
pw  = nn.Conv2d(3, 32, kernel_size=1)                        # pointwise
```

### 9.2 Pooling & Receptive Field

**Pooling** downsamples the spatial dimensions:

| Type | Operation | Effect |
|------|-----------|--------|
| Max pooling | $\max$ over region | Retains strongest activation; translation invariant |
| Average pooling | $\text{mean}$ over region | Smoother; used in global average pooling |
| Global Average Pooling (GAP) | Mean of entire feature map | Collapses spatial dims → vector; replaces FC layer |

**Receptive field:** the region of the original input that affects a given output unit.

After $L$ conv layers with kernel size $k$:

$$\text{RF} = 1 + L(k-1)$$

For $k=3$, $L=5$: $\text{RF} = 11$. Stacking small kernels ($3\times3$) is more efficient than using large ones ($11\times11$): same receptive field, fewer parameters, more nonlinearities.

### 9.3 Classic Architectures

| Architecture | Year | Key Innovation | Top-1 ImageNet |
|-------------|------|---------------|---------------|
| AlexNet | 2012 | ReLU, Dropout, GPU training | 57.1% |
| VGG-16 | 2014 | Deep stacks of 3×3 convs | 71.5% |
| GoogLeNet | 2014 | Inception module (parallel branches) | 74.8% |
| ResNet-50 | 2015 | Skip connections (residuals) | 76.0% |
| DenseNet | 2017 | Dense connections (all layers connected) | 77.1% |
| EfficientNet | 2019 | Compound scaling (depth/width/resolution) | 84.4% |
| ViT | 2020 | Pure Transformer on image patches | 88.5% |

**ResNet residual block:**

$$\mathbf{h}^{(l+1)} = \text{ReLU}\!\left(\mathbf{h}^{(l)} + F(\mathbf{h}^{(l)})\right)$$

$$F(\mathbf{h}) = W_2 \cdot \text{ReLU}(W_1 \cdot \text{BN}(\mathbf{h}))$$

The skip connection enables training of 100+ layer networks by ensuring direct gradient flow.

---

## 10. Recurrent Neural Networks (RNN)

### 10.1 Vanilla RNN

Processes sequences by maintaining a hidden state that summarizes history:

$$\mathbf{h}_t = \tanh\!\left(W_{hh}\mathbf{h}_{t-1} + W_{xh}\mathbf{x}_t + \mathbf{b}_h\right)$$

$$\mathbf{y}_t = W_{hy}\mathbf{h}_t + \mathbf{b}_y$$

| Term | Shape | Meaning |
|------|-------|---------|
| $\mathbf{h}_t$ | $(d_h,)$ | Hidden state at step $t$ — "memory" |
| $\mathbf{x}_t$ | $(d_x,)$ | Input at step $t$ |
| $W_{hh}$ | $(d_h, d_h)$ | Hidden-to-hidden weights |
| $W_{xh}$ | $(d_h, d_x)$ | Input-to-hidden weights |

**Problem:** vanilla RNN suffers from vanishing/exploding gradients through time — difficulty learning long-range dependencies.

### 10.2 LSTM

Long Short-Term Memory uses **gates** to control information flow, maintaining a **cell state** $\mathbf{c}_t$ that can carry information long distances.

$$\mathbf{f}_t = \sigma(W_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \qquad \text{(forget gate)}$$

$$\mathbf{i}_t = \sigma(W_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \qquad \text{(input gate)}$$

$$\tilde{\mathbf{c}}_t = \tanh(W_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) \qquad \text{(candidate cell)}$$

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \qquad \text{(cell update)}$$

$$\mathbf{o}_t = \sigma(W_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \qquad \text{(output gate)}$$

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \qquad \text{(hidden state)}$$

| Gate | Controls | Intuition |
|------|---------|-----------|
| Forget $\mathbf{f}_t$ | What to erase from cell | "Do I need to forget the previous context?" |
| Input $\mathbf{i}_t$ | What new info to write | "Is this new input worth remembering?" |
| Output $\mathbf{o}_t$ | What to expose as hidden state | "What part of memory is relevant to output?" |
| Cell $\mathbf{c}_t$ | Long-term memory | Carries information across many steps |

**Key advantage:** the cell state update $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \ldots$ has additive structure → gradient flows through time without repeated multiplication (similar to ResNet skip connections).

```python
lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2,
               batch_first=True, dropout=0.3)

# x: (batch, seq_len, input_size)
output, (h_n, c_n) = lstm(x)
# output: (batch, seq_len, hidden_size)
# h_n:    (num_layers, batch, hidden_size) — last hidden state
# c_n:    (num_layers, batch, hidden_size) — last cell state
```

### 10.3 GRU

Gated Recurrent Unit — simplified LSTM with fewer gates (no cell state):

$$\mathbf{r}_t = \sigma(W_r [\mathbf{h}_{t-1}, \mathbf{x}_t]) \qquad \text{(reset gate)}$$

$$\mathbf{z}_t = \sigma(W_z [\mathbf{h}_{t-1}, \mathbf{x}_t]) \qquad \text{(update gate)}$$

$$\tilde{\mathbf{h}}_t = \tanh(W_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t]) \qquad \text{(candidate)}$$

$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

| Gate | Meaning |
|------|---------|
| Reset $\mathbf{r}_t$ | How much past hidden state to use when computing candidate |
| Update $\mathbf{z}_t$ | Interpolation between old $\mathbf{h}_{t-1}$ and new $\tilde{\mathbf{h}}_t$ |

**GRU vs LSTM:**

| | GRU | LSTM |
|--|-----|------|
| Parameters | Fewer (~25% less) | More |
| Cell state | No | Yes |
| Performance | Similar | Slightly better on long sequences |
| Speed | Faster | Slower |

---

## 11. Attention & Transformers

### 11.1 Scaled Dot-Product Attention

Given queries $Q$, keys $K$, values $V$:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

| Term | Shape | Meaning |
|------|-------|---------|
| $Q$ | $(N, d_k)$ | Queries — "what am I looking for?" |
| $K$ | $(M, d_k)$ | Keys — "what do I offer as a label?" |
| $V$ | $(M, d_v)$ | Values — "what do I provide as content?" |
| $QK^\top$ | $(N, M)$ | Attention scores — how much query $i$ attends to key $j$ |
| $\sqrt{d_k}$ | scalar | Scaling — prevents large dot products from pushing softmax into saturation |

**Intuition:** for each query (output position), compute similarity with all keys, normalize with softmax to get attention weights, then take a weighted sum of values.

**Why $\sqrt{d_k}$?** The dot product $QK^\top$ has variance proportional to $d_k$. Dividing by $\sqrt{d_k}$ normalizes variance to 1, keeping softmax in a useful gradient regime.

**Self-attention:** $Q = K = V = X$ (the same sequence attends to itself). Each position looks at all other positions.

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (N, M)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    weights = F.softmax(scores, dim=-1)                   # (N, M)
    return weights @ V                                     # (N, d_v)
```

### 11.2 Multi-Head Attention

Run attention $h$ times in parallel with different linear projections, then concatenate:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O$$

$$\text{head}_i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)$$

| Term | Shape | Meaning |
|------|-------|---------|
| $W^Q_i, W^K_i$ | $(d_\text{model}, d_k)$ | Query/Key projection for head $i$ |
| $W^V_i$ | $(d_\text{model}, d_v)$ | Value projection for head $i$ |
| $W^O$ | $(h \cdot d_v, d_\text{model})$ | Output projection |
| $d_k = d_v = d_\text{model}/h$ | typically | Dimension per head |

**Why multiple heads?** Each head can attend to different aspects of the sequence — one might focus on syntax, another on coreference, another on local dependencies.

```python
attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.1, batch_first=True)

# x: (batch, seq_len, 512)
out, weights = attn(query=x, key=x, value=x)   # self-attention
```

### 11.3 Positional Encoding

Attention has no built-in notion of order — two sentences with words shuffled look identical. Positional encoding injects position information.

**Sinusoidal (original Transformer):**

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_\text{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_\text{model}}}\right)$$

| Term | Meaning |
|------|---------|
| $pos$ | Token position in the sequence |
| $i$ | Dimension index |
| $d_\text{model}$ | Embedding dimension |

**Properties:** different frequencies encode different scales of position; the model can attend to relative positions by linear combinations of $\sin$ and $\cos$.

**Learned positional encoding** (used in BERT, GPT): simply train a lookup table $E_\text{pos} \in \mathbb{R}^{T_\text{max} \times d_\text{model}}$.

**RoPE (Rotary Position Embedding):** encodes relative position via rotation in embedding space — used in LLaMA, GPT-NeoX. Naturally extrapolates to longer sequences.

### 11.4 Transformer Architecture

The original Transformer (Vaswani et al., 2017) is an encoder-decoder architecture.

**Encoder block:**

```
Input
  │
  ├── MultiHead Self-Attention
  │         ↓
  │   Add & LayerNorm
  │         ↓
  ├── Feed-Forward (2-layer MLP)
  │         ↓
  └── Add & LayerNorm
          ↓
       Output
```

**Encoder equations:**

$$\mathbf{x}' = \text{LN}(\mathbf{x} + \text{MultiHead}(\mathbf{x}, \mathbf{x}, \mathbf{x}))$$

$$\mathbf{x}'' = \text{LN}(\mathbf{x}' + \text{FFN}(\mathbf{x}'))$$

$$\text{FFN}(\mathbf{x}) = W_2 \cdot \text{ReLU}(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$

**Key design choices:**

| Choice | Reason |
|--------|--------|
| Pre-norm vs Post-norm | Pre-norm (LN before sub-layer) is more stable for deep models |
| FFN width = $4 d_\text{model}$ | Standard; increases model capacity |
| Residual connections | Gradient flow through depth |
| Masking in decoder | Causal mask prevents attending to future tokens |

**Parameter count (1 Transformer block, $d=512$, $h=8$, FFN dim $=2048$):**

| Component | Parameters |
|-----------|-----------|
| MultiHead Attention ($Q, K, V, O$ projections) | $4 \times 512^2 = 1M$ |
| FFN ($W_1, W_2$) | $512 \times 2048 + 2048 \times 512 = 2M$ |
| LayerNorm ($\gamma, \beta$) × 2 | $4 \times 512 \approx 2K$ |
| **Total per block** | **~3M** |

### 11.5 BERT, GPT, and Variants

| Model | Architecture | Training | Use |
|-------|-------------|---------|-----|
| **BERT** | Encoder only | Masked Language Modeling (MLM) + NSP | Understanding (classification, QA) |
| **GPT** | Decoder only | Causal LM (predict next token) | Generation (text, code) |
| **T5** | Encoder-Decoder | Seq2seq (all tasks as text-to-text) | Translation, summarization, QA |
| **LLaMA** | Decoder only | Causal LM (larger scale) | General purpose (efficient) |

**BERT pretraining:**
- **MLM:** mask 15% of tokens randomly, predict them → bidirectional context
- **NSP:** predict if sentence B follows sentence A → now mostly dropped

**GPT pretraining:**
- **Causal LM:** $p(\mathbf{x}) = \prod_t p(x_t \mid x_{<t})$ — predict next token using all previous
- Autoregressive: each token only attends to past (causal mask)

**Fine-tuning paradigm:**

```
Pretrain on large corpus (self-supervised)
       ↓
Fine-tune on task-specific labeled data
       ↓
Evaluate on held-out test set
```

---

## 12. Loss Functions

### 12.1 Regression Losses

**Mean Squared Error (MSE):**

$$\mathcal{L}_\text{MSE} = \frac{1}{N}\sum_i (y_i - \hat{y}_i)^2$$

Penalizes large errors quadratically — sensitive to outliers. Corresponds to MLE under Gaussian noise.

**Mean Absolute Error (MAE):**

$$\mathcal{L}_\text{MAE} = \frac{1}{N}\sum_i |y_i - \hat{y}_i|$$

More robust to outliers. Corresponds to MLE under Laplace noise. Non-differentiable at 0.

**Huber Loss:**

$$\mathcal{L}_\text{Huber} = \begin{cases} \frac{1}{2}(y_i - \hat{y}_i)^2 & \text{if } |y_i - \hat{y}_i| \leq \delta \\ \delta\left(|y_i - \hat{y}_i| - \frac{\delta}{2}\right) & \text{otherwise} \end{cases}$$

Quadratic near zero (smooth gradients), linear for large errors (robust to outliers).

### 12.2 Classification Losses

**Binary Cross-Entropy (BCE):**

$$\mathcal{L}_\text{BCE} = -\frac{1}{N}\sum_i \left[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right]$$

**Categorical Cross-Entropy:**

$$\mathcal{L}_\text{CE} = -\frac{1}{N}\sum_i \log \hat{p}_{y_i}$$

**Label Smoothing:** replace hard labels ($1/0$) with soft labels ($1-\epsilon/\epsilon$) — prevents overconfidence:

$$y_\text{smooth} = (1-\epsilon)\,y_\text{one-hot} + \frac{\epsilon}{K}$$

**Focal Loss** (for class imbalance, from RetinaNet):

$$\mathcal{L}_\text{focal} = -\alpha_t (1-\hat{p}_t)^\gamma \log \hat{p}_t$$

| Term | Meaning |
|------|---------|
| $(1-\hat{p}_t)^\gamma$ | Down-weight easy examples (when $\hat{p}$ is high) |
| $\gamma > 0$ | Focusing parameter (typically 2) |
| $\alpha_t$ | Class weighting to handle imbalance |

### 12.3 Contrastive & Metric Learning Losses

**Contrastive Loss (Siamese Networks):**

$$\mathcal{L} = y \cdot D^2 + (1-y) \cdot \max(0,\, m - D)^2$$

| Term | Meaning |
|------|---------|
| $y = 1$ | Similar pair |
| $y = 0$ | Dissimilar pair |
| $D = \|\mathbf{z}_1 - \mathbf{z}_2\|$ | Distance between embeddings |
| $m$ | Margin — push dissimilar pairs at least $m$ apart |

**Triplet Loss:**

$$\mathcal{L}_\text{triplet} = \max\!\left(0,\, D(a, p) - D(a, n) + m\right)$$

| Term | Meaning |
|------|---------|
| $a$ | Anchor sample |
| $p$ | Positive (same class as anchor) |
| $n$ | Negative (different class) |
| $m$ | Margin |

Encourages: $D(\text{anchor}, \text{positive}) + m < D(\text{anchor}, \text{negative})$

**NT-Xent Loss (SimCLR, contrastive self-supervised):**

$$\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k)/\tau)}$$

| Term | Meaning |
|------|---------|
| $\mathbf{z}_i, \mathbf{z}_j$ | Two augmented views of the same image (positive pair) |
| $\tau$ | Temperature — controls sharpness of distribution |
| $\text{sim}(\mathbf{u},\mathbf{v}) = \mathbf{u}^\top\mathbf{v}/(\|\mathbf{u}\|\|\mathbf{v}\|)$ | Cosine similarity |

---

## 13. Training Techniques

### 13.1 Weight Initialization

Poor initialization → vanishing/exploding activations from the very first forward pass.

**Xavier (Glorot) initialization** — for linear layers with sigmoid/tanh:

$$W \sim \mathcal{U}\!\left[-\sqrt{\frac{6}{n_\text{in}+n_\text{out}}},\; \sqrt{\frac{6}{n_\text{in}+n_\text{out}}}\right]$$

Keeps variance of activations $\approx 1$ across layers.

**He (Kaiming) initialization** — for layers with ReLU:

$$W \sim \mathcal{N}\!\left(0, \sqrt{\frac{2}{n_\text{in}}}\right)$$

Accounts for the fact that ReLU zeroes out ~half the neurons.

| Activation | Initialization | Gain |
|-----------|---------------|------|
| Linear/None | Xavier | 1 |
| Tanh | Xavier | 5/3 |
| Sigmoid | Xavier | 1 |
| ReLU | He (Kaiming) | $\sqrt{2}$ |
| SELU | LeCun | 1 |

```python
# PyTorch applies He init for Conv/Linear by default
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
```

### 13.2 Batch Size & Training Dynamics

**Effect of batch size:**

| Batch Size | Gradient Noise | Generalization | Memory | Typical Use |
|-----------|--------------|---------------|--------|------------|
| Small (8–32) | High | Better (implicit regularization) | Low | Limited memory |
| Medium (128–512) | Moderate | Good | Moderate | Default |
| Large (1K–32K) | Low | Worse (sharp minima) | High | Distributed training |

**Linear scaling rule:** when batch size multiplied by $k$, scale lr by $k$. Keeps gradient noise-to-signal ratio constant.

$$\text{lr}_\text{new} = k \cdot \text{lr}_\text{base}$$

**Gradient accumulation:** simulate large batch size with limited memory.

```python
accumulation_steps = 4   # simulate batch size × 4
optimizer.zero_grad()

for step, (x, y) in enumerate(dataloader):
    loss = criterion(model(x), y) / accumulation_steps
    loss.backward()                          # accumulate gradients

    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 13.3 Mixed Precision Training

Use **FP16** for forward/backward (faster, less memory), **FP32** for weight updates (more precise).

$$\text{Speed}: 2\text{--}3\times \quad \text{Memory}: \sim 2\times \text{ savings}$$

**GradScaler** prevents underflow (FP16 can't represent very small gradients):

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for x, y in dataloader:
    optimizer.zero_grad()

    with autocast():                   # forward in FP16
        pred = model(x)
        loss = criterion(pred, y)

    scaler.scale(loss).backward()      # scale loss to prevent underflow
    scaler.step(optimizer)             # unscale gradients, then update
    scaler.update()                    # adjust scale factor
```

**BF16 (bfloat16):** same range as FP32 (8 exponent bits), lower precision — preferred on A100/H100 GPUs (no overflow risk, no scaling needed).

### 13.4 Transfer Learning & Fine-Tuning

**Transfer learning:** take a model pretrained on a large dataset (ImageNet, large text corpus), adapt it to a downstream task.

**Strategies:**

| Strategy | When to use | What to train |
|----------|-------------|---------------|
| **Feature extraction** | Small target dataset, similar domain | Only new head; freeze backbone |
| **Full fine-tuning** | Large target dataset | All parameters |
| **Partial fine-tuning** | Medium dataset | Unfreeze last $k$ layers |
| **Linear probing** | Very small dataset | Only linear classifier on frozen features |

**Learning rate for fine-tuning:**

Use a **smaller lr** than pretraining (typically $\times 10$ to $\times 100$ smaller) — large updates can destroy pretrained representations.

**Discriminative fine-tuning:** use different learning rates for different layers — lower lr for early layers (which capture general features), higher for later layers (task-specific).

$$\eta^{(l)} = \eta / \gamma^{L-l}, \quad \gamma \approx 2.6$$

**LoRA (Low-Rank Adaptation):** freeze the pretrained weight $W_0$, learn only a low-rank update $\Delta W = BA$:

$$W = W_0 + \Delta W = W_0 + BA, \quad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times d'}$$

| Term | Meaning |
|------|---------|
| $r \ll \min(d, d')$ | Rank — controls # of trainable parameters |
| $B$ | Initialized to zero (so $\Delta W = 0$ at start) |
| $A$ | Initialized from $\mathcal{N}(0, \sigma^2)$ |

LoRA trains only $r(d + d')$ parameters instead of $d \cdot d'$ — typically $0.1\%$ of the full model — enabling efficient fine-tuning of large language models.

```python
# LoRA applied to a linear layer
class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, r=8):
        super().__init__()
        self.W0 = nn.Linear(in_f, out_f, bias=False)
        self.W0.requires_grad_(False)          # freeze
        self.A  = nn.Parameter(torch.randn(r, in_f) * 0.01)
        self.B  = nn.Parameter(torch.zeros(out_f, r))

    def forward(self, x):
        return self.W0(x) + (x @ self.A.T) @ self.B.T
```

```python
from torchvision import models

# Feature extraction: freeze all, train only head
backbone = models.resnet50(pretrained=True)
for param in backbone.parameters():
    param.requires_grad = False

backbone.fc = nn.Linear(2048, num_classes)   # new head
optimizer = torch.optim.Adam(backbone.fc.parameters(), lr=1e-3)

# Partial fine-tuning: unfreeze last block
for param in backbone.layer4.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam([
    {'params': backbone.layer4.parameters(), 'lr': 1e-4},
    {'params': backbone.fc.parameters(),     'lr': 1e-3},
])
```
