# Gaussian Process Regression

Most regression methods pick **one** function that fits the data. Gaussian process regression (GPR) instead keeps a **probability distribution over functions**, conditions it on the data, and returns for every input both a prediction and an honest error bar — large where data is sparse, small where it is dense. It needs no fixed list of features, works well with tens to thousands of data points, and its uncertainty is exactly what robotics needs for learning dynamics, safe exploration, and Bayesian optimization.

Mathematically, GPR is **Gaussian conditioning** — the same operation as the Kalman filter update — applied to the values of an unknown function. This document derives every formula with proof.

**Prerequisites:**

- Multivariate Gaussians, conditioning, and the linear-Gaussian model with proofs — [statistics_probability.md](../basics/statistics_probability.md) §4.3 (Key Properties 1–4)
- Positive (semi-)definite matrices — [linear_algebra.md](../basics/linear_algebra.md) §10; Woodbury and push-through identities — [linear_algebra.md](../basics/linear_algebra.md) §4.6
- MLE and MAP — [statistics_probability.md](../basics/statistics_probability.md) §6.7–6.8; gradients — [multivariable_calculus.md](../basics/multivariable_calculus.md) §6.5

**Notation:**

| Symbol | Meaning |
|--------|---------|
| $X = \{\mathbf{x}_1, \ldots, \mathbf{x}_n\}$, $\mathbf{y} \in \mathbb{R}^n$ | Training inputs and outputs |
| $\mathbf{x}_*$, $f_*$ | A test input and the unknown function value there |
| $k(\mathbf{x}, \mathbf{x}')$ | Kernel (covariance function) |
| $K = K(X, X)$ | $n \times n$ Gram matrix, $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ |
| $\mathbf{k}_* = K(X, \mathbf{x}_*)$, $k_{**} = k(\mathbf{x}_*, \mathbf{x}_*)$ | Train–test covariances, test variance |
| $\sigma_n^2$ | Observation noise variance; $K_y = K + \sigma_n^2 I$ |

## Table of Contents

1. [From Linear Regression to Distributions over Functions](#1-from-linear-regression-to-distributions-over-functions)
   - 1.1 [Bayesian Linear Regression](#11-bayesian-linear-regression)
   - 1.2 [The Predictive Distribution](#12-the-predictive-distribution)
   - 1.3 [The Kernel Trick: From Weights to Functions](#13-the-kernel-trick-from-weights-to-functions)
2. [Gaussian Processes](#2-gaussian-processes)
   - 2.1 [Definition](#21-definition)
   - 2.2 [Consistency](#22-consistency)
   - 2.3 [Sampling Functions from a GP](#23-sampling-functions-from-a-gp)
3. [Kernels](#3-kernels)
   - 3.1 [Valid Kernels Are Positive Semi-Definite Functions](#31-valid-kernels-are-positive-semi-definite-functions)
   - 3.2 [Common Kernels](#32-common-kernels)
   - 3.3 [Building New Kernels — Closure Properties](#33-building-new-kernels--closure-properties)
   - 3.4 [Proof That the Squared-Exponential Kernel Is Valid](#34-proof-that-the-squared-exponential-kernel-is-valid)
   - 3.5 [Stationary Kernels & Bochner's Theorem](#35-stationary-kernels--bochners-theorem)
4. [GP Regression](#4-gp-regression)
   - 4.1 [The Joint Distribution of Training and Test Outputs](#41-the-joint-distribution-of-training-and-test-outputs)
   - 4.2 [The Predictive Distribution](#42-the-predictive-distribution)
   - 4.3 [Properties of the Posterior](#43-properties-of-the-posterior)
   - 4.4 [Worked Example](#44-worked-example)
5. [Learning Hyperparameters](#5-learning-hyperparameters)
   - 5.1 [The Marginal Likelihood](#51-the-marginal-likelihood)
   - 5.2 [Occam's Razor Built In](#52-occams-razor-built-in)
   - 5.3 [The Gradient](#53-the-gradient)
   - 5.4 [Optimization in Practice](#54-optimization-in-practice)
6. [Computation](#6-computation)
   - 6.1 [The Cholesky Algorithm](#61-the-cholesky-algorithm)
   - 6.2 [Numerical Issues](#62-numerical-issues)
   - 6.3 [Scaling to Large Datasets](#63-scaling-to-large-datasets)
7. [Connections](#7-connections)
   - 7.1 [Kernel Ridge Regression](#71-kernel-ridge-regression)
   - 7.2 [GPs and the Kalman Filter](#72-gps-and-the-kalman-filter)
   - 7.3 [Bayesian Optimization](#73-bayesian-optimization)
8. [Robotics Applications](#8-robotics-applications)
   - 8.1 [Learning Dynamics Residuals](#81-learning-dynamics-residuals)
   - 8.2 [Other Uses](#82-other-uses)
   - 8.3 [Implementation](#83-implementation)

---

## 1. From Linear Regression to Distributions over Functions

### 1.1 Bayesian Linear Regression

Model the function as a linear combination of features $\boldsymbol\phi(\mathbf{x}) \in \mathbb{R}^d$ (e.g. $\boldsymbol\phi(x) = (1, x, x^2)$):

$$f(\mathbf{x}) = \boldsymbol\phi(\mathbf{x})^\top\mathbf{w}, \qquad y_i = f(\mathbf{x}_i) + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma_n^2) \text{ i.i.d.}$$

Stack the features into the **design matrix** $\Phi \in \mathbb{R}^{n \times d}$ (row $i$ = $\boldsymbol\phi(\mathbf{x}_i)^\top$), so $\mathbf{y} = \Phi\mathbf{w} + \boldsymbol\varepsilon$. Instead of a single best $\mathbf{w}$, put a **prior** on it: $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \Sigma_p)$.

**Theorem (posterior over weights).**

$$\mathbf{w} \mid \mathbf{y} \sim \mathcal{N}\big(\bar{\mathbf{w}},\ A^{-1}\big), \qquad A = \sigma_n^{-2}\Phi^\top\Phi + \Sigma_p^{-1}, \qquad \bar{\mathbf{w}} = \sigma_n^{-2}A^{-1}\Phi^\top\mathbf{y}$$

*Proof.* This is the linear-Gaussian model (Key Property 4) with $\mathbf{x} \to \mathbf{w}$, $\boldsymbol\mu = \mathbf{0}$, $P = \Sigma_p$, $H = \Phi$, $\mathbf{c} = \mathbf{0}$, $R = \sigma_n^2 I$. Its information form (iv) gives posterior precision $\Sigma_p^{-1} + \Phi^\top(\sigma_n^2 I)^{-1}\Phi = A$ and mean $A^{-1}\big(\Sigma_p^{-1}\mathbf{0} + \Phi^\top\sigma_n^{-2}\mathbf{y}\big) = \bar{\mathbf{w}}$. $\blacksquare$

**The posterior mean is ridge regression.** With $\Sigma_p = \sigma_p^2 I$, $\bar{\mathbf{w}} = (\Phi^\top\Phi + \lambda I)^{-1}\Phi^\top\mathbf{y}$ with $\lambda = \sigma_n^2/\sigma_p^2$. Since the posterior is Gaussian, mean = mode, so this is also the **MAP estimate** — the prior plays the role of the L2 penalty ([linear_algebra.md](../basics/linear_algebra.md) §13.5, [statistics_probability.md](../basics/statistics_probability.md) §6.8).

### 1.2 The Predictive Distribution

At a test input, $f_* = \boldsymbol\phi_*^\top\mathbf{w}$ with $\boldsymbol\phi_* = \boldsymbol\phi(\mathbf{x}_*)$ — an affine map of the Gaussian posterior. By Key Property 1:

$$f_* \mid \mathbf{y} \sim \mathcal{N}\big(\boldsymbol\phi_*^\top\bar{\mathbf{w}},\ \boldsymbol\phi_*^\top A^{-1}\boldsymbol\phi_*\big)$$

The variance is large in directions of feature space the data has not constrained. A noisy observation $y_* = f_* + \varepsilon_*$ has variance larger by $\sigma_n^2$.

**Limitation:** the model can only represent functions in the span of the chosen features. Choosing features is hard; using very many is expensive ($A$ is $d \times d$). The next step removes both problems.

### 1.3 The Kernel Trick: From Weights to Functions

**Theorem (function-space form).** Define the kernel $k(\mathbf{x}, \mathbf{x}') = \boldsymbol\phi(\mathbf{x})^\top\Sigma_p\,\boldsymbol\phi(\mathbf{x}')$, so $K = \Phi\Sigma_p\Phi^\top$, $\mathbf{k}_* = \Phi\Sigma_p\boldsymbol\phi_*$, $k_{**} = \boldsymbol\phi_*^\top\Sigma_p\boldsymbol\phi_*$. Then the predictive distribution of §1.2 equals

$$\boxed{\ \mathbb{E}[f_* \mid \mathbf{y}] = \mathbf{k}_*^\top(K + \sigma_n^2 I)^{-1}\mathbf{y}, \qquad \operatorname{Var}[f_* \mid \mathbf{y}] = k_{**} - \mathbf{k}_*^\top(K + \sigma_n^2 I)^{-1}\mathbf{k}_*\ }$$

*Proof — mean.* The push-through identity ([linear_algebra.md](../basics/linear_algebra.md) §4.6) with $P = \Sigma_p$, $H = \Phi$, $R = \sigma_n^2 I$ reads

$$\Sigma_p\Phi^\top(\Phi\Sigma_p\Phi^\top + \sigma_n^2 I)^{-1} = (\Sigma_p^{-1} + \sigma_n^{-2}\Phi^\top\Phi)^{-1}\Phi^\top\sigma_n^{-2} = \sigma_n^{-2}A^{-1}\Phi^\top$$

Hence $\boldsymbol\phi_*^\top\bar{\mathbf{w}} = \boldsymbol\phi_*^\top\sigma_n^{-2}A^{-1}\Phi^\top\mathbf{y} = \boldsymbol\phi_*^\top\Sigma_p\Phi^\top(K + \sigma_n^2 I)^{-1}\mathbf{y} = \mathbf{k}_*^\top(K + \sigma_n^2 I)^{-1}\mathbf{y}$.

*Proof — variance.* Woodbury with $A = \Sigma_p^{-1}$, $U = \Phi^\top$, $C = \sigma_n^{-2}I$, $V = \Phi$:

$$A^{-1} = (\Sigma_p^{-1} + \sigma_n^{-2}\Phi^\top\Phi)^{-1} = \Sigma_p - \Sigma_p\Phi^\top(\sigma_n^2 I + \Phi\Sigma_p\Phi^\top)^{-1}\Phi\Sigma_p$$

Sandwiching with $\boldsymbol\phi_*$ gives $\boldsymbol\phi_*^\top A^{-1}\boldsymbol\phi_* = k_{**} - \mathbf{k}_*^\top(K + \sigma_n^2 I)^{-1}\mathbf{k}_*$. $\blacksquare$

> **Key insight:** the features appear **only through inner products** $k(\mathbf{x}, \mathbf{x}')$. So we can skip features entirely and choose $k$ directly — even kernels corresponding to infinitely many features (§3.4). The cost now scales with the number of data points $n$, not the number of features $d$.

| | Weight space (§1.1–1.2) | Function space (§1.3) |
|---|---|---|
| Inverts | $d \times d$ matrix $A$ | $n \times n$ matrix $K + \sigma_n^2 I$ |
| Needs | Explicit features $\boldsymbol\phi$ | Only the kernel $k$ |
| Best when | Few features, many data | Few data, rich (even infinite) features |

---

## 2. Gaussian Processes

### 2.1 Definition

A **Gaussian process** is a collection of random variables $\{f(\mathbf{x})\}_{\mathbf{x} \in \mathcal{X}}$, any **finite** number of which have a joint Gaussian distribution. It is fully specified by a **mean function** and a **covariance function**:

$$f \sim \mathcal{GP}\big(m(\mathbf{x}),\ k(\mathbf{x}, \mathbf{x}')\big), \qquad m(\mathbf{x}) = \mathbb{E}[f(\mathbf{x})], \qquad k(\mathbf{x}, \mathbf{x}') = \operatorname{Cov}\big(f(\mathbf{x}), f(\mathbf{x}')\big)$$

Meaning: for any inputs $\mathbf{x}_1, \ldots, \mathbf{x}_n$,

$$\begin{bmatrix} f(\mathbf{x}_1) \\ \vdots \\ f(\mathbf{x}_n) \end{bmatrix} \sim \mathcal{N}\!\left(\begin{bmatrix} m(\mathbf{x}_1) \\ \vdots \\ m(\mathbf{x}_n) \end{bmatrix},\ \begin{bmatrix} k(\mathbf{x}_1, \mathbf{x}_1) & \cdots & k(\mathbf{x}_1, \mathbf{x}_n) \\ \vdots & \ddots & \vdots \\ k(\mathbf{x}_n, \mathbf{x}_1) & \cdots & k(\mathbf{x}_n, \mathbf{x}_n) \end{bmatrix}\right)$$

Think of it as an **infinite-dimensional Gaussian vector** indexed by $\mathbf{x}$ instead of by an integer. The mean function is usually taken as $0$ (after subtracting the data mean, or by putting prior knowledge in a parametric mean). The kernel encodes everything else: smoothness, length scale, periodicity, amplitude.

Bayesian linear regression is a GP with $m = 0$ and $k(\mathbf{x}, \mathbf{x}') = \boldsymbol\phi(\mathbf{x})^\top\Sigma_p\boldsymbol\phi(\mathbf{x}')$ — check: $\operatorname{Cov}(\boldsymbol\phi^\top\mathbf{w}, \boldsymbol\phi'^\top\mathbf{w}) = \boldsymbol\phi^\top\Sigma_p\boldsymbol\phi'$.

### 2.2 Consistency

For the definition to make sense, the finite-dimensional distributions must agree with each other: the distribution of $(f(\mathbf{x}_1), f(\mathbf{x}_2))$ computed from the 2-point specification must equal its marginal computed from any larger specification such as $(f(\mathbf{x}_1), f(\mathbf{x}_2), f(\mathbf{x}_3))$.

**Theorem.** The specification "$\mathcal{N}$ with mean $m(\cdot)$ and covariance $k(\cdot, \cdot)$ evaluated at the chosen points" is consistent.

*Proof.* Marginalizing a Gaussian keeps the corresponding sub-vector of the mean and sub-block of the covariance (Key Property 2). The sub-block of $[k(\mathbf{x}_i, \mathbf{x}_j)]$ for a subset of points is exactly the matrix the specification assigns to that subset. $\blacksquare$

By the Kolmogorov extension theorem, consistent finite-dimensional distributions define a stochastic process. Consistency is also what makes GP regression work: predictions at test points do not depend on which other unobserved points we imagine.

### 2.3 Sampling Functions from a GP

To draw a random function on a grid $\mathbf{x}_1, \ldots, \mathbf{x}_N$: compute $K$, factor $K + \epsilon I = LL^\top$ (Cholesky, with tiny "jitter" $\epsilon \approx 10^{-8}$ for numerical stability), and set

$$\mathbf{f} = L\mathbf{z}, \qquad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, I) \quad\Rightarrow\quad \mathbf{f} \sim \mathcal{N}(\mathbf{0}, LL^\top)$$

(Key Property 1). Plotting several samples is the best way to see what prior a kernel expresses.

---

## 3. Kernels

### 3.1 Valid Kernels Are Positive Semi-Definite Functions

A function $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ can be a covariance function **iff** it is

1. **symmetric:** $k(\mathbf{x}, \mathbf{x}') = k(\mathbf{x}', \mathbf{x})$, and
2. **positive semi-definite:** for every finite set of points, the Gram matrix $K$ is PSD — equivalently $\sum_{i,j}c_ic_jk(\mathbf{x}_i, \mathbf{x}_j) \geq 0$ for all real $c_1, \ldots, c_n$.

*Why necessary:* $\sum_{i,j}c_ic_jk(\mathbf{x}_i, \mathbf{x}_j) = \operatorname{Var}\big(\sum_i c_if(\mathbf{x}_i)\big) \geq 0$. *Why sufficient:* any symmetric PSD matrix is a valid Gaussian covariance, and the family is consistent by §2.2.

An invalid "kernel" produces Gram matrices with negative eigenvalues — negative variances — and the Cholesky factorization fails.

### 3.2 Common Kernels

Write $r = \|\mathbf{x} - \mathbf{x}'\|$. Hyperparameters: **signal variance** $\sigma_f^2$ (vertical scale of the functions) and **length scale** $\ell$ (how far apart inputs must be before function values become unrelated).

| Kernel | $k(\mathbf{x}, \mathbf{x}')$ | Sample functions |
|--------|----------------------------|------------------|
| Squared exponential (SE, RBF) | $\sigma_f^2\exp\!\Big(-\dfrac{r^2}{2\ell^2}\Big)$ | Infinitely differentiable — very smooth |
| Matérn $\nu = \tfrac12$ (exponential) | $\sigma_f^2\exp\!\Big(-\dfrac{r}{\ell}\Big)$ | Continuous, nowhere differentiable (rough, like Brownian motion) |
| Matérn $\nu = \tfrac32$ | $\sigma_f^2\Big(1 + \dfrac{\sqrt3\,r}{\ell}\Big)\exp\!\Big(-\dfrac{\sqrt3\,r}{\ell}\Big)$ | Once differentiable |
| Matérn $\nu = \tfrac52$ | $\sigma_f^2\Big(1 + \dfrac{\sqrt5\,r}{\ell} + \dfrac{5r^2}{3\ell^2}\Big)\exp\!\Big(-\dfrac{\sqrt5\,r}{\ell}\Big)$ | Twice differentiable — a common default for physical systems |
| Periodic | $\sigma_f^2\exp\!\Big(-\dfrac{2\sin^2(\pi r/p)}{\ell^2}\Big)$ | Exactly periodic with period $p$ |
| Linear | $\sigma_b^2 + \sigma_v^2\,\mathbf{x}^\top\mathbf{x}'$ | Straight lines — Bayesian linear regression |
| White noise | $\sigma_n^2\,\mathbf{1}[\mathbf{x} = \mathbf{x}']$ | Independent noise at every point |

**Automatic relevance determination (ARD):** use one length scale per input dimension, $r^2 = \sum_d (x_d - x'_d)^2/\ell_d^2$. A learned $\ell_d \to \infty$ means input $d$ is irrelevant — built-in feature selection.

### 3.3 Building New Kernels — Closure Properties

Let $k_1, k_2$ be valid kernels with Gram matrices $K_1, K_2$.

| Construction | Valid because |
|-------------|---------------|
| $c\,k_1$, $c > 0$ | $cK_1 \succeq 0$ |
| $k_1 + k_2$ | $K_1 + K_2 \succeq 0$ (sum of PSD matrices) — models a sum of independent functions |
| $k_1 \cdot k_2$ | $K_1 \circ K_2 \succeq 0$ by the Schur product theorem (below) |
| $g(\mathbf{x})\,k_1(\mathbf{x}, \mathbf{x}')\,g(\mathbf{x}')$ for any function $g$ | Gram matrix $D_gK_1D_g$ with $D_g = \operatorname{diag}(g(\mathbf{x}_i))$ — a congruence |
| $p(k_1)$, polynomial with non-negative coefficients | Sums and products of the above |
| $\exp(k_1)$ | Limit of the series $\sum_j k_1^j/j!$; pointwise limits of PSD kernels are PSD (the PSD cone is closed) |
| $k_1(\boldsymbol\psi(\mathbf{x}), \boldsymbol\psi(\mathbf{x}'))$ for any map $\boldsymbol\psi$ | Gram matrix of $k_1$ at the points $\boldsymbol\psi(\mathbf{x}_i)$ |

**Schur product theorem.** If $A, B \succeq 0$, then the element-wise (Hadamard) product $A \circ B \succeq 0$.

*Proof.* Eigendecompose $B = \sum_k \lambda_k\mathbf{q}_k\mathbf{q}_k^\top = \sum_k \mathbf{b}_k\mathbf{b}_k^\top$ with $\mathbf{b}_k = \sqrt{\lambda_k}\,\mathbf{q}_k$ ($\lambda_k \geq 0$). For one rank-1 term, $(A \circ \mathbf{b}\mathbf{b}^\top)_{ij} = A_{ij}b_ib_j = (D_{\mathbf{b}}AD_{\mathbf{b}})_{ij}$ with $D_{\mathbf{b}} = \operatorname{diag}(\mathbf{b})$, and $\mathbf{c}^\top D_{\mathbf{b}}AD_{\mathbf{b}}\mathbf{c} = (D_{\mathbf{b}}\mathbf{c})^\top A(D_{\mathbf{b}}\mathbf{c}) \geq 0$. So $A \circ B = \sum_k D_{\mathbf{b}_k}AD_{\mathbf{b}_k}$ is a sum of PSD matrices. $\blacksquare$

**Modeling with kernel algebra** (examples):

- Smooth trend + seasonal pattern: $k_{\text{SE}} + k_{\text{per}}$.
- Periodic pattern whose shape slowly changes: $k_{\text{per}} \cdot k_{\text{SE}}$.
- Signal + measurement noise: $k_{\text{signal}} + k_{\text{white}}$ (this is where $\sigma_n^2 I$ in $K_y$ comes from).

### 3.4 Proof That the Squared-Exponential Kernel Is Valid

**Theorem.** $k(\mathbf{x}, \mathbf{x}') = \exp\!\big(-\|\mathbf{x} - \mathbf{x}'\|^2/2\ell^2\big)$ is a valid kernel on $\mathbb{R}^d$.

*Proof.* Expand the square, $\|\mathbf{x} - \mathbf{x}'\|^2 = \|\mathbf{x}\|^2 - 2\mathbf{x}^\top\mathbf{x}' + \|\mathbf{x}'\|^2$:

$$k(\mathbf{x}, \mathbf{x}') = \underbrace{e^{-\|\mathbf{x}\|^2/2\ell^2}}_{g(\mathbf{x})}\cdot\exp\!\Big(\frac{\mathbf{x}^\top\mathbf{x}'}{\ell^2}\Big)\cdot\underbrace{e^{-\|\mathbf{x}'\|^2/2\ell^2}}_{g(\mathbf{x}')}$$

- $\mathbf{x}^\top\mathbf{x}'/\ell^2$ is a valid (linear) kernel: its Gram matrix is $XX^\top/\ell^2 \succeq 0$.
- Its exponential is valid (closure under $\exp$, §3.3).
- Multiplying by $g(\mathbf{x})\,g(\mathbf{x}')$ keeps validity (closure under congruence). $\blacksquare$

**Infinitely many features.** The series $\exp(\mathbf{x}^\top\mathbf{x}'/\ell^2) = \sum_j (\mathbf{x}^\top\mathbf{x}')^j/(\ell^{2j}j!)$ contains monomial features of every degree. A GP with the SE kernel is Bayesian linear regression with **infinitely many** basis functions — computable only because of the kernel trick (§1.3).

### 3.5 Stationary Kernels & Bochner's Theorem

A kernel is **stationary** if it depends only on $\boldsymbol\tau = \mathbf{x} - \mathbf{x}'$: $k(\mathbf{x}, \mathbf{x}') = k(\boldsymbol\tau)$. All kernels in §3.2 except the linear one are stationary — the prior looks the same everywhere in input space.

**Bochner's theorem.** A continuous stationary $k(\boldsymbol\tau)$ is positive semi-definite **iff** it is the Fourier transform of a non-negative finite measure — its **spectral density** $S(\boldsymbol\omega) \geq 0$:

$$k(\boldsymbol\tau) = \int S(\boldsymbol\omega)\,e^{i\boldsymbol\omega^\top\boldsymbol\tau}\,d\boldsymbol\omega$$

*Proof of the "if" direction.* For any points and real coefficients,

$$\sum_{i,j}c_ic_jk(\mathbf{x}_i - \mathbf{x}_j) = \int S(\boldsymbol\omega)\sum_{i,j}c_ic_je^{i\boldsymbol\omega^\top\mathbf{x}_i}e^{-i\boldsymbol\omega^\top\mathbf{x}_j}\,d\boldsymbol\omega = \int S(\boldsymbol\omega)\,\Big|\sum_i c_ie^{i\boldsymbol\omega^\top\mathbf{x}_i}\Big|^2 d\boldsymbol\omega \geq 0 \qquad \blacksquare$$

(The "only if" direction needs more measure theory and is omitted.)

**Reading kernels in frequency.** The spectral density says which frequencies the random functions contain ([signals_systems.md](../engineering_math/signals_systems.md) §3). The SE kernel is a Gaussian, so its spectrum is a Gaussian too — high frequencies are suppressed extremely fast, hence the very smooth samples. Matérn spectra decay only polynomially, $S(\omega) \propto (2\nu/\ell^2 + \omega^2)^{-(\nu + d/2)}$, leaving more high-frequency content and rougher samples. A shorter length scale $\ell$ means a wider spectrum: faster wiggles.

---

## 4. GP Regression

### 4.1 The Joint Distribution of Training and Test Outputs

**Model:** $f \sim \mathcal{GP}(0, k)$, observations $y_i = f(\mathbf{x}_i) + \varepsilon_i$ with i.i.d. $\varepsilon_i \sim \mathcal{N}(0, \sigma_n^2)$ independent of $f$.

**Claim.** The training outputs and the test function value are jointly Gaussian:

$$\begin{bmatrix} \mathbf{y} \\ f_* \end{bmatrix} \sim \mathcal{N}\!\left(\mathbf{0},\ \begin{bmatrix} K + \sigma_n^2 I & \mathbf{k}_* \\ \mathbf{k}_*^\top & k_{**} \end{bmatrix}\right)$$

*Proof.* By the GP definition, $(\mathbf{f}, f_*) = (f(\mathbf{x}_1), \ldots, f(\mathbf{x}_n), f(\mathbf{x}_*))$ is Gaussian with covariance $\begin{bmatrix} K & \mathbf{k}_* \\ \mathbf{k}_*^\top & k_{**} \end{bmatrix}$. The noise $\boldsymbol\varepsilon \sim \mathcal{N}(\mathbf{0}, \sigma_n^2 I)$ is independent of it, so $(\mathbf{f}, f_*, \boldsymbol\varepsilon)$ is jointly Gaussian with block-diagonal covariance, and $(\mathbf{y}, f_*) = (\mathbf{f} + \boldsymbol\varepsilon, f_*)$ is an affine map of it. Key Property 1: the noise adds $\sigma_n^2 I$ to the $\mathbf{y}$ block only, and leaves the cross-covariance $\operatorname{Cov}(\mathbf{f} + \boldsymbol\varepsilon, f_*) = \mathbf{k}_*$ unchanged. $\blacksquare$

### 4.2 The Predictive Distribution

**Theorem (GP posterior predictive).**

$$f_* \mid X, \mathbf{y}, \mathbf{x}_* \sim \mathcal{N}(\bar{f}_*, \mathbb{V}[f_*]), \qquad \boxed{\ \bar{f}_* = \mathbf{k}_*^\top(K + \sigma_n^2 I)^{-1}\mathbf{y}, \qquad \mathbb{V}[f_*] = k_{**} - \mathbf{k}_*^\top(K + \sigma_n^2 I)^{-1}\mathbf{k}_*\ }$$

*Proof.* Condition the joint Gaussian of §4.1 on $\mathbf{y}$ (Key Property 3 with $a = f_*$, $b = \mathbf{y}$): $\boldsymbol\mu_{a|b} = 0 + \mathbf{k}_*^\top(K + \sigma_n^2 I)^{-1}(\mathbf{y} - \mathbf{0})$ and $\Sigma_{a|b} = k_{**} - \mathbf{k}_*^\top(K + \sigma_n^2 I)^{-1}\mathbf{k}_*$. ($K + \sigma_n^2 I \succ 0$ when $\sigma_n^2 > 0$, so the inverse exists.) $\blacksquare$

This matches the weight-space result of §1.3, as it must. For a noisy **observation** at $\mathbf{x}_*$, add the noise: $\mathbb{V}[y_*] = \mathbb{V}[f_*] + \sigma_n^2$.

**Many test points at once.** For test inputs $X_*$, the same conditioning gives a joint Gaussian:

$$\bar{\mathbf{f}}_* = K(X_*, X)K_y^{-1}\mathbf{y}, \qquad \operatorname{Cov}(\mathbf{f}_*) = K(X_*, X_*) - K(X_*, X)K_y^{-1}K(X, X_*)$$

The full covariance (not just the diagonal) is what you need to draw coherent sample functions from the posterior.

**Non-zero prior mean** $m(\mathbf{x})$: apply the result to $f - m$, giving $\bar{f}_* = m(\mathbf{x}_*) + \mathbf{k}_*^\top K_y^{-1}(\mathbf{y} - \mathbf{m}_X)$ with $\mathbf{m}_X = (m(\mathbf{x}_i))_i$. Far from the data the prediction reverts to the prior mean.

### 4.3 Properties of the Posterior

**1. The mean is a weighted sum of kernels centered on the data.** With $\boldsymbol\alpha = K_y^{-1}\mathbf{y}$,

$$\bar{f}(\mathbf{x}_*) = \sum_{i=1}^n \alpha_i\,k(\mathbf{x}_i, \mathbf{x}_*)$$

— a finite expansion even though the model has infinitely many features (a representer theorem, §7.1). It is also **linear in $\mathbf{y}$**: $\bar{f}_* = \mathbf{h}(\mathbf{x}_*)^\top\mathbf{y}$ with weights $\mathbf{h} = K_y^{-1}\mathbf{k}_*$ — a "linear smoother".

**2. The variance does not depend on $\mathbf{y}$** — only on where the data are (same as the Kalman filter covariance, both inherited from Key Property 3). Uncertainty can be planned before collecting data: choose experiments where the predicted variance is largest (active learning).

**3. Data never increases uncertainty:** $\mathbb{V}[f_*] \leq k_{**}$, because $\mathbf{k}_*^\top K_y^{-1}\mathbf{k}_* \geq 0$ ($K_y^{-1} \succ 0$). Far from all data, $\mathbf{k}_* \to \mathbf{0}$ (for decaying kernels) and the variance returns to the prior $k_{**} = \sigma_f^2$.

**4. Noise-free interpolation.** With $\sigma_n^2 = 0$ (and $K$ invertible), at a training point $\mathbf{x}_* = \mathbf{x}_j$ the vector $\mathbf{k}_*$ is the $j$-th column of $K$, so $K^{-1}\mathbf{k}_* = \mathbf{e}_j$. Then $\bar{f}(\mathbf{x}_j) = \mathbf{e}_j^\top\mathbf{y} = y_j$ and $\mathbb{V} = K_{jj} - \mathbf{k}_*^\top\mathbf{e}_j = 0$: the posterior passes **exactly** through the data with zero uncertainty there.

**5. The posterior is again a GP**, with mean $\bar{f}(\cdot)$ and covariance function

$$k_{\text{post}}(\mathbf{x}, \mathbf{x}') = k(\mathbf{x}, \mathbf{x}') - K(\mathbf{x}, X)\,K_y^{-1}\,K(X, \mathbf{x}')$$

so it can be conditioned on more data later — sequential updating, just like a filter.

### 4.4 Worked Example

SE kernel with $\sigma_f = 1$, $\ell = 1$; noise $\sigma_n^2 = 0.1$; two training points $(x, y) = (0, 1)$ and $(1, 0)$.

**Gram matrix:** $k(0, 1) = e^{-1/2} = 0.6065$, so

$$K_y = \begin{bmatrix} 1.1 & 0.6065 \\ 0.6065 & 1.1 \end{bmatrix}, \qquad \det K_y = 1.21 - 0.3679 = 0.8421, \qquad \boldsymbol\alpha = K_y^{-1}\mathbf{y} = \frac{1}{0.8421}\begin{bmatrix} 1.1 \\ -0.6065 \end{bmatrix} = \begin{bmatrix} 1.3062 \\ -0.7202 \end{bmatrix}$$

**Prediction at $x_* = 0.5$:** $\mathbf{k}_* = (e^{-1/8}, e^{-1/8}) = (0.8825, 0.8825)$.

$$\bar{f}_* = 0.8825\,(1.3062 - 0.7202) = 0.5171$$

$$\mathbf{k}_*^\top K_y^{-1}\mathbf{k}_* = 0.8825^2\cdot\frac{1.1 + 1.1 - 2(0.6065)}{0.8421} = 0.9127 \quad\Rightarrow\quad \mathbb{V}[f_*] = 1 - 0.9127 = 0.0873, \quad \text{std} = 0.295$$

**Compare three test locations:**

| $x_*$ | Mean | Std | Comment |
|-------|------|-----|---------|
| $0$ (a training input) | $0.869$ | $0.295$ | Pulled toward $y = 1$ but not all the way — the data are noisy |
| $0.5$ (between the points) | $0.517$ | $0.295$ | Smooth interpolation |
| $2$ (outside the data) | $-0.260$ | $0.783$ | Drifting back toward the prior mean $0$; uncertainty approaching the prior std $1$ |

**Actual example — what the length scale does.** Same two data points and noise, three length scales:

| $\ell$ | Mean / std at $x_* = 0.5$ | Mean / std at $x_* = 2$ | Interpretation |
|--------|--------------------------|------------------------|----------------|
| $0.3$ | $0.226$ / $0.942$ | $0.000$ / $1.000$ | Points are "far apart" relative to $\ell$: almost no information between them, and none at all at $x = 2$ |
| $1$ | $0.517$ / $0.295$ | $-0.260$ / $0.783$ | Smooth interpolation, moderate extrapolation |
| $3$ | $0.482$ / $0.222$ | $-0.045$ / $0.431$ | Very smooth: confident even well outside the data |

A short length scale says "the function can change quickly, so a measurement only tells you about its immediate neighborhood"; a long one says "the function is smooth, so each measurement informs a wide region". Section 5 shows how to let the data choose.

---

## 5. Learning Hyperparameters

The kernel's hyperparameters $\boldsymbol\theta$ (e.g. $\sigma_f$, $\ell$, $\sigma_n$) control the fit. They are learned by maximizing the probability of the observed data.

### 5.1 The Marginal Likelihood

**Theorem.**

$$\boxed{\ \log p(\mathbf{y} \mid X, \boldsymbol\theta) = -\tfrac12\,\mathbf{y}^\top K_y^{-1}\mathbf{y} - \tfrac12\log\det K_y - \tfrac{n}{2}\log 2\pi\ }$$

*Proof.* The function values are integrated out ("marginal"): $p(\mathbf{y}) = \int p(\mathbf{y} \mid \mathbf{f})\,p(\mathbf{f})\,d\mathbf{f}$. With $\mathbf{f} \sim \mathcal{N}(\mathbf{0}, K)$ and $\mathbf{y} = \mathbf{f} + \boldsymbol\varepsilon$, Key Property 4(ii) (with $H = I$, $R = \sigma_n^2 I$) gives $\mathbf{y} \sim \mathcal{N}(\mathbf{0}, K + \sigma_n^2 I)$ without computing any integral. Take the log of the Gaussian density. $\blacksquare$

### 5.2 Occam's Razor Built In

| Term | Meaning | Prefers |
|------|---------|---------|
| $-\tfrac12\mathbf{y}^\top K_y^{-1}\mathbf{y}$ | **Data fit** | Flexible models that explain $\mathbf{y}$ |
| $-\tfrac12\log\det K_y$ | **Complexity penalty** (volume of the prior over data sets) | Simple models |
| $-\tfrac{n}{2}\log 2\pi$ | Constant | — |

A tiny length scale fits any data (good fit term) but spreads its prior probability over a huge variety of possible data sets (bad complexity term); a huge length scale is the opposite. The maximum balances the two automatically — no validation set is required.

### 5.3 The Gradient

Two matrix-calculus lemmas, for a symmetric invertible $K(\theta)$ depending on a scalar $\theta$:

**Lemma 1.** $\dfrac{\partial K^{-1}}{\partial\theta} = -K^{-1}\dfrac{\partial K}{\partial\theta}K^{-1}$.

*Proof.* Differentiate $KK^{-1} = I$: $\dot{K}K^{-1} + K\,\tfrac{\partial K^{-1}}{\partial\theta} = 0$; multiply on the left by $K^{-1}$. $\blacksquare$

**Lemma 2 (Jacobi's formula).** $\dfrac{\partial}{\partial\theta}\log\det K = \operatorname{tr}\Big(K^{-1}\dfrac{\partial K}{\partial\theta}\Big)$.

*Proof.* $\det(K + \delta K) = \det K\,\det(I + K^{-1}\delta K)$. For a small matrix $E$ with eigenvalues $\epsilon_i$, $\det(I + E) = \prod_i(1 + \epsilon_i) = 1 + \sum_i\epsilon_i + O(\|E\|^2) = 1 + \operatorname{tr}E + O(\|E\|^2)$. Hence $\log\det(K + \delta K) - \log\det K = \log\big(1 + \operatorname{tr}(K^{-1}\delta K) + \ldots\big) = \operatorname{tr}(K^{-1}\delta K) + O(\|\delta K\|^2)$. $\blacksquare$

**Theorem (gradient of the log marginal likelihood).** With $\boldsymbol\alpha = K_y^{-1}\mathbf{y}$:

$$\frac{\partial}{\partial\theta_j}\log p(\mathbf{y} \mid X, \boldsymbol\theta) = \tfrac12\,\mathbf{y}^\top K_y^{-1}\frac{\partial K_y}{\partial\theta_j}K_y^{-1}\mathbf{y} - \tfrac12\operatorname{tr}\Big(K_y^{-1}\frac{\partial K_y}{\partial\theta_j}\Big) = \tfrac12\operatorname{tr}\Big(\big(\boldsymbol\alpha\boldsymbol\alpha^\top - K_y^{-1}\big)\frac{\partial K_y}{\partial\theta_j}\Big)$$

*Proof.* Lemma 1 on the first term of §5.1, Lemma 2 on the second; then $\boldsymbol\alpha^\top\dot{K}\boldsymbol\alpha = \operatorname{tr}(\boldsymbol\alpha\boldsymbol\alpha^\top\dot{K})$ (cyclic property of the trace). $\blacksquare$

**Kernel derivatives** for the SE kernel $k = \sigma_f^2\exp(-r^2/2\ell^2)$, using log-parameters (which keep $\sigma_f, \ell, \sigma_n$ positive during optimization):

| Parameter | $\partial K_y / \partial(\cdot)$, entry-wise |
|-----------|------------------------------------------|
| $\log\sigma_f$ | $2\,k(\mathbf{x}_i, \mathbf{x}_j)$ |
| $\log\ell$ | $k(\mathbf{x}_i, \mathbf{x}_j)\,r_{ij}^2/\ell^2$ |
| $\log\sigma_n$ | $2\sigma_n^2\,\delta_{ij}$ |

Each gradient costs $O(n^3)$ once $K_y^{-1}$ is available ($O(n^2)$ per extra parameter).

### 5.4 Optimization in Practice

- Optimize over $\log\boldsymbol\theta$ with L-BFGS ([optimization.md](../basics/optimization.md) §8.2) using the analytic gradient.
- The marginal likelihood is **non-convex**: use several random restarts and keep the best.
- Typical local optima: "everything is noise" (large $\sigma_n$, long $\ell$) vs "fit everything" (tiny $\sigma_n$, short $\ell$). Sensible initial values (e.g. $\ell$ ≈ typical input spacing, $\sigma_n$ ≈ known sensor noise) and weak priors on $\boldsymbol\theta$ (giving MAP-II instead of ML-II) help.
- **Normalize** inputs and outputs (zero mean, unit variance) before fitting.

---

## 6. Computation

### 6.1 The Cholesky Algorithm

Never form $K_y^{-1}$ explicitly. Factor once and reuse:

| Step | Operation | Cost |
|------|-----------|------|
| 1 | $L = \operatorname{chol}(K + \sigma_n^2 I)$, so $K_y = LL^\top$ | $n^3/3$ |
| 2 | $\boldsymbol\alpha = L^\top \backslash (L \backslash \mathbf{y})$ (two triangular solves) | $n^2$ |
| 3 | Mean: $\bar{f}_* = \mathbf{k}_*^\top\boldsymbol\alpha$ | $n$ per test point |
| 4 | $\mathbf{v} = L \backslash \mathbf{k}_*$; variance: $\mathbb{V}[f_*] = k_{**} - \mathbf{v}^\top\mathbf{v}$ | $n^2$ per test point |
| 5 | $\log p(\mathbf{y}) = -\tfrac12\mathbf{y}^\top\boldsymbol\alpha - \sum_i\log L_{ii} - \tfrac{n}{2}\log 2\pi$ | $n$ |

*Why step 4:* $\mathbf{k}_*^\top K_y^{-1}\mathbf{k}_* = \mathbf{k}_*^\top L^{-\top}L^{-1}\mathbf{k}_* = \|L^{-1}\mathbf{k}_*\|^2$. *Why step 5:* $\det K_y = \det L\,\det L^\top = \prod_i L_{ii}^2$, so $\tfrac12\log\det K_y = \sum_i\log L_{ii}$ — computing the determinant directly would overflow or underflow.

**Check with §4.4:** $L = \begin{bmatrix} 1.0488 & 0 \\ 0.5783 & 0.8750 \end{bmatrix}$ ($1.0488^2 = 1.1$, $1.0488 \times 0.5783 = 0.6065$ ✓), and the log marginal likelihood comes out as $-2.405$.

### 6.2 Numerical Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Cholesky fails ("not positive definite") | Nearly duplicate inputs or very long $\ell$ make $K$ almost singular | Add jitter $10^{-8}$–$10^{-6}$ to the diagonal; keep $\sigma_n$ bounded away from $0$ |
| Negative predicted variances | Rounding error in $k_{**} - \mathbf{v}^\top\mathbf{v}$ | Clip at $0$ |
| Hyperparameters run off to extremes | Flat or multimodal likelihood | Priors on $\boldsymbol\theta$, bounds, normalized data |

### 6.3 Scaling to Large Datasets

Exact GPR costs $O(n^3)$ time and $O(n^2)$ memory — fine up to roughly $10^4$ points.

| Approach | Idea | Cost |
|----------|------|------|
| **Inducing points** (SoR, FITC, variational SGPR) | Summarize the data through $m \ll n$ pseudo-inputs $Z$; approximate $K \approx K_{XZ}K_{ZZ}^{-1}K_{ZX}$ (Nyström), a rank-$m$ matrix | $O(nm^2)$ — via Woodbury and the determinant lemma ([linear_algebra.md](../basics/linear_algebra.md) §4.6) |
| **Random Fourier features** | Sample frequencies from the spectral density (Bochner, §3.5) to build explicit features; then do Bayesian linear regression (§1.1) | $O(nD^2)$ for $D$ features |
| **State-space form** (1D inputs, e.g. time) | Matérn kernels correspond to linear SDEs; run a Kalman filter + smoother (§7.2) | $O(n)$ |
| Local GPs | Fit separate GPs to regions of input space | Depends on partition |

---

## 7. Connections

### 7.1 Kernel Ridge Regression

**Theorem.** The GP posterior mean equals the kernel ridge regression (KRR) solution with regularization $\lambda = \sigma_n^2$.

*Proof.* KRR minimizes $\sum_i (y_i - f(\mathbf{x}_i))^2 + \lambda\|f\|_{\mathcal{H}}^2$ over the reproducing-kernel Hilbert space $\mathcal{H}$ of $k$ — the space spanned by functions $k(\mathbf{x}_i, \cdot)$, with inner product $\langle k(\mathbf{x}, \cdot), k(\mathbf{x}', \cdot)\rangle = k(\mathbf{x}, \mathbf{x}')$ and the **reproducing property** $f(\mathbf{x}) = \langle f, k(\mathbf{x}, \cdot)\rangle$.

*Step 1 — representer theorem.* Split $f = f_\parallel + f_\perp$, with $f_\parallel \in \operatorname{span}\{k(\mathbf{x}_i, \cdot)\}$ and $f_\perp$ orthogonal to it. By the reproducing property, $f(\mathbf{x}_i) = \langle f_\parallel + f_\perp, k(\mathbf{x}_i, \cdot)\rangle = f_\parallel(\mathbf{x}_i)$, so the data term ignores $f_\perp$, while $\|f\|^2 = \|f_\parallel\|^2 + \|f_\perp\|^2$. The minimizer therefore has $f_\perp = 0$: $f = \sum_i\alpha_ik(\mathbf{x}_i, \cdot)$.

*Step 2 — solve.* Then $f(X) = K\boldsymbol\alpha$ and $\|f\|^2 = \boldsymbol\alpha^\top K\boldsymbol\alpha$, so the objective is $\|\mathbf{y} - K\boldsymbol\alpha\|^2 + \lambda\boldsymbol\alpha^\top K\boldsymbol\alpha$. Its gradient is $-2K(\mathbf{y} - K\boldsymbol\alpha) + 2\lambda K\boldsymbol\alpha = 2K\big((K + \lambda I)\boldsymbol\alpha - \mathbf{y}\big)$, which vanishes at $\boldsymbol\alpha = (K + \lambda I)^{-1}\mathbf{y}$. The prediction $f(\mathbf{x}_*) = \mathbf{k}_*^\top(K + \lambda I)^{-1}\mathbf{y}$ is the GP mean with $\lambda = \sigma_n^2$. $\blacksquare$

KRR gives the same curve; the GP adds the **variance** and a principled way to set $\lambda$ and the kernel (§5).

### 7.2 GPs and the Kalman Filter

**Theorem.** The Matérn-½ (exponential) kernel $k(t, t') = \sigma^2 e^{-|t - t'|/\ell}$ is exactly the covariance of the stationary Ornstein–Uhlenbeck process

$$\dot{x} = -\frac{1}{\ell}x + w(t), \qquad w \text{ white noise with intensity } q = \frac{2\sigma^2}{\ell}$$

*Proof.* *Stationary variance:* for $\dot{x} = ax + w$ the variance $P(t)$ obeys $\dot{P} = 2aP + q$ (the scalar Lyapunov differential equation). At stationarity $0 = -\tfrac{2}{\ell}P + q$, so $P = q\ell/2 = \sigma^2$. *Covariance:* for $\tau \geq 0$, solving the ODE from time $t$ ([ode_linear_systems.md](../engineering_math/ode_linear_systems.md) §5.2) gives $x(t + \tau) = e^{-\tau/\ell}x(t) + \int_t^{t+\tau}e^{-(t + \tau - s)/\ell}w(s)\,ds$; the integral involves only future noise, independent of $x(t)$. Hence $\operatorname{Cov}(x(t + \tau), x(t)) = e^{-\tau/\ell}\operatorname{Var}(x(t)) = \sigma^2e^{-\tau/\ell}$. $\blacksquare$

**Consequence.** Sampled at times $t_k$ with gaps $\Delta_k$, the process is the linear-Gaussian state-space model

$$x_k = e^{-\Delta_k/\ell}x_{k-1} + w_k, \qquad w_k \sim \mathcal{N}\big(0,\ \sigma^2(1 - e^{-2\Delta_k/\ell})\big), \qquad y_k = x_k + v_k$$

(the noise variance is exactly what keeps the stationary variance at $\sigma^2$). So GP regression in time with this kernel is **Kalman filtering** ([kalman_filter.md](kalman_filter.md)) — the filter gives $p(x_k \mid y_{1:k})$, and a backward Rauch–Tung–Striebel smoothing pass gives the full GP posterior $p(x_k \mid y_{1:n})$ — in $O(n)$ instead of $O(n^3)$. Matérn-3/2 and 5/2 kernels correspond likewise to 2- and 3-dimensional linear SDEs (position, velocity, acceleration). The two documents describe the **same Gaussian conditioning**, one in batch form and one recursively.

### 7.3 Bayesian Optimization

To optimize an expensive black-box function (a gait parameter set evaluated on hardware, a controller gain tuned by experiment), fit a GP to the evaluations so far and choose the next point by maximizing an **acquisition function** that trades exploitation (high mean) against exploration (high uncertainty). With posterior mean $\mu(\mathbf{x})$, std $s(\mathbf{x})$, and best value so far $f^+$ (maximization):

**Upper confidence bound:** $a_{\text{UCB}}(\mathbf{x}) = \mu(\mathbf{x}) + \beta\,s(\mathbf{x})$.

**Expected improvement:** $a_{\text{EI}}(\mathbf{x}) = \mathbb{E}\big[\max(f(\mathbf{x}) - f^+, 0)\big] = (\mu - f^+)\,\Phi(Z) + s\,\varphi(Z)$, $Z = \dfrac{\mu - f^+}{s}$, where $\Phi$, $\varphi$ are the standard normal CDF and PDF.

*Proof of the EI formula.* With $f = \mu + su$, $u \sim \mathcal{N}(0, 1)$, the improvement is positive when $u > -Z$:

$$\mathbb{E}[\max(f - f^+, 0)] = \int_{-Z}^{\infty}(\mu - f^+ + su)\,\varphi(u)\,du = (\mu - f^+)\big(1 - \Phi(-Z)\big) + s\int_{-Z}^{\infty}u\,\varphi(u)\,du$$

Since $\varphi'(u) = -u\varphi(u)$, $\int_{-Z}^{\infty}u\varphi(u)\,du = \varphi(-Z) = \varphi(Z)$, and $1 - \Phi(-Z) = \Phi(Z)$. $\blacksquare$

---

## 8. Robotics Applications

### 8.1 Learning Dynamics Residuals

A physics model $\dot{\mathbf{x}} = \mathbf{f}_{\text{nom}}(\mathbf{x}, \mathbf{u})$ is never exact (friction, cable forces, aerodynamic effects). Learn only the **residual**:

$$\dot{\mathbf{x}} = \mathbf{f}_{\text{nom}}(\mathbf{x}, \mathbf{u}) + \mathbf{g}(\mathbf{x}, \mathbf{u}), \qquad g_j \sim \mathcal{GP}(0, k_j) \text{ for each output dimension } j$$

Training targets are measured accelerations minus model predictions. The GP returns a correction **and its uncertainty**; a controller (e.g. GP-MPC) can tighten safety margins where the learned model is uncertain and explore cautiously. Keeping the physics model as the prior mean means the GP only has to learn the small, smooth part — far fewer data are needed.

**Actual example — learning a joint's friction torque.** A joint's friction (unknown to the model) is $\tau_f(v) = 0.5\tanh(5v) + 0.1v$. We measure it at 11 velocities $v = -1, -0.8, \ldots, 1$ rad/s with sensor noise $\sigma_n = 0.03$ N·m (known from the torque sensor), and fit $\sigma_f$, $\ell$ by maximizing the marginal likelihood (§5). Result: $\sigma_f = 0.475$, $\ell = 0.451$.

| Query velocity | GP mean ± 2σ | True friction | Region |
|----------------|-------------|---------------|--------|
| $0.1$ | $0.208 \pm 0.044$ | $0.241$ | Inside the data, steep part of the curve |
| $0.3$ | $0.484 \pm 0.044$ | $0.483$ | Inside the data |
| $0.7$ | $0.576 \pm 0.046$ | $0.569$ | Inside the data |
| $1.5$ | $0.344 \pm 0.622$ | $0.650$ | Beyond the data — uncertainty grows |
| $3.0$ | $0.000 \pm 0.950$ | $0.800$ | Far beyond — back to the prior |

Inside the training range the prediction is accurate to a few hundredths of a N·m, with an error bar that contains the truth. Outside it, the GP does **not** pretend: the mean drifts to the prior and the error bar widens to cover the true value. A controller using this model knows to be cautious above 1 rad/s, where no data were collected — exactly the behavior a neural network fit would not provide by itself. (When the noise level was also left free on such a small data set, the optimizer drove $\sigma_n \to 0$ and became overconfident — the practical reason to fix or put a prior on $\sigma_n$, §5.4.)

### 8.2 Other Uses

| Application | How the GP is used |
|-------------|-------------------|
| Bayesian optimization of gaits / controller gains | §7.3 — few hardware trials |
| Terrain and elevation mapping | Height as a GP over $(x, y)$; variance shows where to sense next |
| Spatial field mapping (gas, temperature, signal strength) | GP over space; active sensing at maximum variance |
| Safe exploration | Only visit states where the GP's confidence bound guarantees safety |
| Sensor calibration and time series | Smooth drift as a GP over time (state-space form, §7.2) |

### 8.3 Implementation

```python
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

def se_kernel(A, B, sf=1.0, ell=1.0):
    d2 = ((A[:, None, :] - B[None, :, :])**2).sum(-1)
    return sf**2 * np.exp(-0.5 * d2 / ell**2)

def gp_predict(X, y, Xs, sf=1.0, ell=1.0, sn=0.1):
    Ky = se_kernel(X, X, sf, ell) + sn**2 * np.eye(len(X))
    L = np.linalg.cholesky(Ky)                          # §6.1 step 1
    alpha = cho_solve((L, True), y)                     # step 2
    Ks = se_kernel(X, Xs, sf, ell)
    mean = Ks.T @ alpha                                  # step 3
    v = solve_triangular(L, Ks, lower=True)              # step 4
    var = np.clip(sf**2 - (v**2).sum(0), 0, None)
    lml = -0.5 * y @ alpha - np.log(np.diag(L)).sum() - 0.5 * len(X) * np.log(2*np.pi)
    return mean, var, lml

X = np.array([[0.], [1.]]); y = np.array([1., 0.])
mean, var, lml = gp_predict(X, y, np.array([[0.5]]), sn=np.sqrt(0.1))
# mean ≈ 0.517, var ≈ 0.087, lml ≈ -2.405  (matches §4.4, §6.1)
```

For real projects, libraries such as GPyTorch, scikit-learn's `GaussianProcessRegressor`, or GPflow provide kernels, hyperparameter optimization, and sparse approximations.

---

## Summary — What to Master First

| Priority | Topic | Because |
|----------|-------|---------|
| 1 | Gaussian conditioning ([statistics_probability.md](../basics/statistics_probability.md) §4.3) and the predictive formulas (§4.1–4.2) | The entire method is one conditioning step |
| 2 | Kernels: PSD requirement, common choices, length scale meaning (§3.1–3.2) | The kernel is the model |
| 3 | Bayesian linear regression and the kernel trick (§1) | Explains what a GP is and why kernels replace features |
| 4 | Marginal likelihood and its gradient (§5) | How hyperparameters are actually set |
| 5 | Cholesky implementation (§6.1) | Numerically correct code |
| 6 | Connections to KRR, Kalman filtering, Bayesian optimization (§7) | Ties GPR to the rest of the curriculum |

**Companion documents:** [kalman_filter.md](kalman_filter.md) (the same conditioning, done recursively in time) · [statistics_probability.md](../basics/statistics_probability.md) (Gaussian properties with proofs; MLE/MAP) · [linear_algebra.md](../basics/linear_algebra.md) (PD matrices, Schur complement, Woodbury) · [signals_systems.md](../engineering_math/signals_systems.md) (spectra, for Bochner's theorem)
