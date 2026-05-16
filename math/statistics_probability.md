# Statistics & Probability for AI

Probability theory is the language of uncertainty. Every AI model — from a classifier outputting a softmax distribution to a diffusion model sampling from a Gaussian — is built on these foundations.

## Table of Contents

1. [Probability vs Statistics](#1-probability-vs-statistics)
2. [Probability Basics](#2-probability-basics)
   - 2.1 [Sample Space & Events](#21-sample-space--events)
   - 2.2 [Axioms of Probability](#22-axioms-of-probability)
   - 2.3 [Conditional Probability](#23-conditional-probability)
   - 2.4 [Independence](#24-independence)
   - 2.5 [Bayes' Theorem](#25-bayes-theorem)
3. [Random Variables](#3-random-variables)
   - 3.1 [Discrete vs Continuous](#31-discrete-vs-continuous)
   - 3.2 [PMF, PDF, CDF](#32-pmf-pdf-cdf)
   - 3.3 [Joint, Marginal, Conditional Distributions](#33-joint-marginal-conditional-distributions)
4. [Expectation & Variance](#4-expectation--variance)
   - 4.1 [Expected Value](#41-expected-value)
   - 4.2 [Variance & Standard Deviation](#42-variance--standard-deviation)
   - 4.3 [Covariance & Correlation](#43-covariance--correlation)
5. [Key Distributions](#5-key-distributions)
   - 5.1 [Bernoulli, Binomial, Multinoulli, Multinomial](#51-bernoulli-binomial-multinoulli-multinomial)
   - 5.2 [Gaussian (Normal)](#52-gaussian-normal)
   - 5.3 [Multivariate Gaussian](#53-multivariate-gaussian)
   - 5.4 [Categorical & Softmax](#54-categorical--softmax)
   - 5.5 [Poisson Distribution](#55-poisson-distribution)
   - 5.6 [Gamma Distribution](#56-gamma-distribution)
   - 5.7 [Beta & Dirichlet Distributions](#57-beta--dirichlet-distributions)
6. [Information Theory](#6-information-theory)
   - 6.1 [Entropy](#61-entropy)
   - 6.2 [Cross-Entropy](#62-cross-entropy)
   - 6.3 [KL Divergence & Total Variation Distance](#63-kl-divergence--total-variation-distance)
   - 6.4 [Mutual Information](#64-mutual-information)
7. [Estimation](#7-estimation)
   - 7.1 [Bias — Definition](#71-bias--definition)
   - 7.2 [Biased vs Unbiased Estimation](#72-biased-vs-unbiased-estimation)
   - 7.3 [Sample Variance & the N−1 Correction](#73-sample-variance--the-n1-correction)
   - 7.4 [Is Unbiased Always Better?](#74-is-unbiased-always-better)
   - 7.5 [Bias, Variance, MSE — and their Relationship](#75-bias-variance-mse--and-their-relationship)
   - 7.6 [MLE vs Sample Variance in the Gaussian](#76-mle-vs-sample-variance-in-the-gaussian)
   - 7.7 [Maximum Likelihood Estimation (MLE)](#77-maximum-likelihood-estimation-mle)
   - 7.8 [Maximum A Posteriori (MAP)](#78-maximum-a-posteriori-map)
   - 7.9 [Conjugate Prior](#79-conjugate-prior)
8. [Variation & Model Fit](#8-variation--model-fit)
   - 8.1 [Total, Explained, Unexplained Variation](#81-total-explained-unexplained-variation)
   - 8.2 [Coefficient of Determination R²](#82-coefficient-of-determination-r)
9. [Hypothesis Testing](#9-hypothesis-testing)
   - 9.1 [P-value](#91-p-value)
   - 9.2 [Confidence Interval](#92-confidence-interval)
   - 9.3 [Likelihood-Ratio Test](#93-likelihood-ratio-test)
10. [Sampling & Monte Carlo](#10-sampling--monte-carlo)
    - 10.1 [Monte Carlo Estimation](#101-monte-carlo-estimation)
    - 10.2 [Importance Sampling](#102-importance-sampling)
11. [Key Theorems](#11-key-theorems)
    - 11.1 [Law of Large Numbers](#111-law-of-large-numbers)
    - 11.2 [Central Limit Theorem](#112-central-limit-theorem)
12. [AI Applications](#12-ai-applications)
    - 12.1 [Loss Functions as Likelihoods](#121-loss-functions-as-likelihoods)
    - 12.2 [Bayesian Deep Learning](#122-bayesian-deep-learning)
    - 12.3 [Variational Autoencoders (VAE)](#123-variational-autoencoders-vae)
    - 12.4 [Diffusion Models](#124-diffusion-models)

---

## 1. Probability vs Statistics

These two fields are deeply related but answer different questions.

| | **Probability** | **Statistics** |
|---|---|---|
| **Direction** | Model → Data | Data → Model |
| **Question** | "Given a model, what data will I see?" | "Given data, what model generated it?" |
| **Known** | The distribution $p(x \mid \theta)$ | The observations $\{x_1, \ldots, x_N\}$ |
| **Goal** | Compute $P(\text{event})$, expectations | Estimate $\theta$, test hypotheses |
| **Example** | Fair coin: $P(\text{5 heads in 10 flips}) = ?$ | I flipped 7 heads — is the coin fair? |

**Probability is deductive:** we reason forward from known rules.

$$\text{Model} \xrightarrow{\text{probability}} \text{Data}$$

> "If the die is fair, the probability of rolling a 6 is 1/6."

**Statistics is inductive:** we reason backward from observed data.

$$\text{Data} \xrightarrow{\text{statistics}} \text{Model}$$

> "I rolled the die 100 times and got 6 exactly 25 times. Is it loaded?"

**Why the distinction matters in AI:**

- **Training a model** is statistics: you observe data $\mathcal{D}$ and estimate parameters $\hat{\theta}$.
- **Running a model** is probability: you use the learned $\hat{\theta}$ to compute $p(y \mid x, \hat{\theta})$.
- **Bayesian methods** blend both: start with a prior (probability), observe data (statistics), and update to a posterior (probability).

```
                 ┌─────────────┐
    prior P(θ)   │             │  posterior P(θ│D)
   ────────────→ │  Bayes Rule │ ──────────────→
    data D       │             │
                 └─────────────┘
   (statistics supplies D; probability computes the posterior)
```

---

## 2. Probability Basics

### 2.1 Sample Space & Events

| Term | Symbol | Meaning |
|------|--------|---------|
| Sample space | $\Omega$ | Set of all possible outcomes |
| Event | $A \subseteq \Omega$ | A subset of outcomes we care about |
| Probability | $P(A)$ | Number in $[0,1]$ representing likelihood of $A$ |

**Example — rolling a die:**

$$\Omega = \{1, 2, 3, 4, 5, 6\}, \quad A = \{\text{even}\} = \{2, 4, 6\}, \quad P(A) = \frac{3}{6} = 0.5$$

**Set operations on events:**

| Operation | Notation | Meaning |
|-----------|----------|---------|
| Union | $A \cup B$ | $A$ or $B$ (at least one occurs) |
| Intersection | $A \cap B$ | $A$ and $B$ (both occur) |
| Complement | $A^c$ | $A$ does not occur |

### 2.2 Axioms of Probability

All of probability theory follows from three axioms (Kolmogorov):

$$P(A) \geq 0 \quad \text{(non-negativity)}$$

$$P(\Omega) = 1 \quad \text{(normalization)}$$

$$P(A \cup B) = P(A) + P(B) \quad \text{if } A \cap B = \emptyset \quad \text{(additivity)}$$

**Derived rules:**

$$P(A^c) = 1 - P(A)$$

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

### 2.3 Conditional Probability

The probability of $A$ given that $B$ has occurred:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

| Term | Meaning |
|------|---------|
| $P(A \mid B)$ | Updated belief about $A$ after observing $B$ |
| $P(A \cap B)$ | Both $A$ and $B$ occur |
| $P(B)$ | Normalizer — restricts the sample space to $B$ |

**Chain rule:**

$$P(A \cap B) = P(A \mid B)\,P(B) = P(B \mid A)\,P(A)$$

$$P(A_1, A_2, \ldots, A_n) = P(A_1)\,P(A_2 \mid A_1)\,P(A_3 \mid A_1, A_2)\cdots$$

**Law of total probability:**

$$P(A) = \sum_i P(A \mid B_i)\,P(B_i) \quad \text{where } B_i \text{ partition } \Omega$$

### 2.4 Independence

$A$ and $B$ are **independent** if knowing one gives no information about the other:

$$A \perp B \iff P(A \cap B) = P(A)\,P(B) \iff P(A \mid B) = P(A)$$

**Conditional independence:** $A \perp B \mid C$ means $A$ and $B$ are independent given $C$:

$$P(A \cap B \mid C) = P(A \mid C)\,P(B \mid C)$$

**In AI:** the Naive Bayes classifier assumes all features are conditionally independent given the class label — a strong but often effective assumption.

### 2.5 Bayes' Theorem

The most important formula in probabilistic AI:

$$P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}$$

| Term | Name | Meaning |
|------|------|---------|
| $P(A \mid B)$ | **Posterior** | Updated belief about $A$ after seeing data $B$ |
| $P(B \mid A)$ | **Likelihood** | How probable is data $B$ if $A$ is true |
| $P(A)$ | **Prior** | Belief about $A$ before seeing data |
| $P(B)$ | **Evidence** | Normalizing constant — ensures posterior sums to 1 |

$$\underbrace{P(\theta \mid \mathcal{D})}_{\text{posterior}} = \frac{\underbrace{P(\mathcal{D} \mid \theta)}_{\text{likelihood}} \cdot \underbrace{P(\theta)}_{\text{prior}}}{\underbrace{P(\mathcal{D})}_{\text{evidence}}}$$

> **Intuition:** start with a prior belief about model parameters $\theta$, observe data $\mathcal{D}$, update to a posterior. The likelihood tells you how well $\theta$ explains $\mathcal{D}$.

```python
import numpy as np

# Bayes' theorem: P(disease | positive test)
p_disease   = 0.01    # prior: 1% of population has the disease
p_pos_given_disease  = 0.99   # sensitivity
p_pos_given_healthy  = 0.05   # false positive rate

p_positive = p_pos_given_disease * p_disease + p_pos_given_healthy * (1 - p_disease)
p_disease_given_pos = (p_pos_given_disease * p_disease) / p_positive

print(f"P(disease | positive) = {p_disease_given_pos:.3f}")  # ~0.167
```

---

## 3. Random Variables

### 3.1 Discrete vs Continuous

| Type | Values | Described by |
|------|--------|-------------|
| **Discrete** | Countable (integers) | Probability Mass Function (PMF) |
| **Continuous** | Uncountable (real numbers) | Probability Density Function (PDF) |

### 3.2 PMF, PDF, CDF

**PMF** (discrete): $p(x) = P(X = x)$, must satisfy $\sum_x p(x) = 1$

**PDF** (continuous): $f(x)$ where $P(a \leq X \leq b) = \int_a^b f(x)\,dx$, must satisfy $\int_{-\infty}^{\infty} f(x)\,dx = 1$

> Note: $f(x)$ is **not** a probability — it is a density. $P(X = x) = 0$ for any single point in a continuous distribution.

**CDF** (both): $F(x) = P(X \leq x)$

$$F(x) = \sum_{t \leq x} p(t) \quad \text{(discrete)}, \qquad F(x) = \int_{-\infty}^{x} f(t)\,dt \quad \text{(continuous)}$$

```python
from scipy import stats

# Discrete: Binomial
X = stats.binom(n=10, p=0.3)
print(X.pmf(3))    # P(X=3)
print(X.cdf(3))    # P(X≤3)

# Continuous: Gaussian
X = stats.norm(loc=0, scale=1)
print(X.pdf(0))    # density at 0 = 1/√(2π) ≈ 0.399
print(X.cdf(0))    # P(X≤0) = 0.5
```

### 3.3 Joint, Marginal, Conditional Distributions

**Joint distribution:** $p(x, y)$ — probability of $X = x$ AND $Y = y$ simultaneously.

**Marginal distribution:** integrate (or sum) out the other variable:

$$p(x) = \int p(x, y)\,dy \qquad \text{(continuous)}$$

$$p(x) = \sum_y p(x, y) \qquad \text{(discrete)}$$

**Conditional distribution:**

$$p(y \mid x) = \frac{p(x, y)}{p(x)}$$

| Relationship | Formula |
|-------------|---------|
| Joint → Marginal | Sum/integrate out unwanted variable |
| Joint → Conditional | Divide by the marginal |
| Conditional + Marginal → Joint | $p(x,y) = p(y\mid x)\,p(x)$ |

---

## 4. Expectation & Variance

### 4.1 Expected Value

The **expected value** (mean) is the probability-weighted average of all outcomes:

$$\mathbb{E}[X] = \sum_x x\,p(x) \qquad \text{(discrete)}$$

$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x\,f(x)\,dx \qquad \text{(continuous)}$$

**Key properties:**

$$\mathbb{E}[aX + b] = a\,\mathbb{E}[X] + b \qquad \text{(linearity)}$$

$$\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y] \qquad \text{(always, even if } X,Y \text{ dependent)}$$

$$\mathbb{E}[XY] = \mathbb{E}[X]\,\mathbb{E}[Y] \qquad \text{(only if } X \perp Y\text{)}$$

**Law of Total Expectation:**

$$\mathbb{E}[X] = \mathbb{E}_Y\!\left[\mathbb{E}[X \mid Y]\right]$$

> **In AI:** the policy gradient $\mathbb{E}_\pi[\nabla_\theta \log\pi_\theta \cdot A]$ is an expectation estimated by averaging over sampled trajectories.

### 4.2 Variance & Standard Deviation

Variance measures how spread out the distribution is around the mean:

$$\text{Var}(X) = \mathbb{E}\!\left[(X - \mathbb{E}[X])^2\right] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

$$\text{Std}(X) = \sqrt{\text{Var}(X)}$$

| Term | Meaning |
|------|---------|
| $\text{Var}(X) = 0$ | $X$ is deterministic — no randomness |
| Large $\text{Var}(X)$ | Outcomes spread far from the mean |
| $\text{Var}(aX) = a^2\,\text{Var}(X)$ | Scaling multiplies variance by $a^2$ |
| $\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y)$ | Only if $X \perp Y$ |

```python
import numpy as np

x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])

print(x.mean())    # 5.0
print(x.var())     # 4.0
print(x.std())     # 2.0
```

### 4.3 Covariance & Correlation

**Covariance** measures how two variables move together:

$$\text{Cov}(X, Y) = \mathbb{E}\!\left[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])\right] = \mathbb{E}[XY] - \mathbb{E}[X]\,\mathbb{E}[Y]$$

| $\text{Cov}(X,Y)$ | Meaning |
|-------------------|---------|
| $> 0$ | $X$ and $Y$ tend to increase together |
| $= 0$ | No linear relationship (not necessarily independent) |
| $< 0$ | When $X$ increases, $Y$ tends to decrease |

**Correlation** normalizes covariance to $[-1, 1]$:

$$\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\text{Std}(X)\,\text{Std}(Y)}$$

**Covariance matrix** for a random vector $\mathbf{x} \in \mathbb{R}^n$:

$$\Sigma = \text{Cov}(\mathbf{x}) = \mathbb{E}\!\left[(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^\top\right] \in \mathbb{R}^{n \times n}$$

$$\Sigma_{ij} = \text{Cov}(x_i, x_j), \qquad \Sigma_{ii} = \text{Var}(x_i)$$

The covariance matrix is always **symmetric** and **positive semi-definite**.

```python
X = np.random.randn(100, 3)   # 100 samples, 3 features

cov = np.cov(X.T)             # (3, 3) covariance matrix
corr = np.corrcoef(X.T)       # (3, 3) correlation matrix
```

---

## 5. Key Distributions

### 5.1 Bernoulli, Binomial, Multinoulli, Multinomial

**Bernoulli($p$):** single binary trial — coin flip, click / no-click.

$$P(X = 1) = p, \quad P(X = 0) = 1-p$$

$$\mathbb{E}[X] = p, \quad \text{Var}(X) = p(1-p)$$

**Binomial($n, p$):** number of successes in $n$ independent Bernoulli trials.

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

$$\mathbb{E}[X] = np, \quad \text{Var}(X) = np(1-p)$$

**Multinoulli (Categorical):** generalization of Bernoulli to $K > 2$ outcomes.  
Single trial, one of $K$ outcomes with probabilities $p_1, \ldots, p_K$ where $\sum_k p_k = 1$.

$$P(X = k) = p_k, \quad k \in \{1, \ldots, K\}$$

$$\mathbb{E}[\mathbf{1}[X=k]] = p_k$$

> Bernoulli is Multinoulli with $K = 2$.

**Multinomial($n, \mathbf{p}$):** generalization of Binomial to $K$ categories.  
$n$ trials, count how many land in each category — e.g., rolling a $K$-sided die $n$ times.

$$P(X_1 = k_1, \ldots, X_K = k_K) = \frac{n!}{k_1!\cdots k_K!}\,p_1^{k_1}\cdots p_K^{k_K}, \quad \sum_j k_j = n$$

$$\mathbb{E}[X_k] = np_k, \quad \text{Var}(X_k) = np_k(1-p_k)$$

| Distribution | Trials | Outcomes | Result |
|-------------|--------|----------|--------|
| Bernoulli | 1 | 2 | 0 or 1 |
| Binomial | $n$ | 2 | count of 1s |
| Multinoulli | 1 | $K$ | which category |
| Multinomial | $n$ | $K$ | count per category |

**In AI:**
- Bernoulli: binary classification label
- Multinoulli: class label in multiclass classification
- Multinomial: word counts in a document (bag-of-words)
- Softmax outputs a Multinoulli parameter vector $\mathbf{p}$

```python
import numpy as np

# Multinomial: roll K=4 sided die n=10 times
p = [0.1, 0.4, 0.3, 0.2]  # probabilities for each face
counts = np.random.multinomial(n=10, pvals=p)
# e.g. [1, 4, 3, 2] — counts for each category
```

### 5.2 Gaussian (Normal)

The most important distribution in AI and statistics:

$$\mathcal{N}(\mu, \sigma^2): \quad f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

| Parameter | Meaning |
|-----------|---------|
| $\mu$ | Mean — center of the bell curve |
| $\sigma^2$ | Variance — width of the bell curve |
| $\sigma$ | Standard deviation |

**Standard normal:** $\mathcal{N}(0, 1)$ — mean 0, variance 1.

**68-95-99.7 rule:**

$$P(\mu - \sigma \leq X \leq \mu + \sigma) \approx 68\%$$
$$P(\mu - 2\sigma \leq X \leq \mu + 2\sigma) \approx 95\%$$
$$P(\mu - 3\sigma \leq X \leq \mu + 3\sigma) \approx 99.7\%$$

**Why Gaussians dominate AI:**
- Central Limit Theorem — sums of random variables converge to Gaussian
- Maximum entropy distribution given fixed mean and variance
- Closed-form KL divergence and marginals
- Weight initialization, noise in diffusion models, VAE latent space

```python
import numpy as np
from scipy import stats

x = np.random.normal(loc=0.0, scale=1.0, size=1000)

z = stats.norm(0, 1)
z.pdf(0)        # 0.3989
z.cdf(1.96)     # 0.975 — 95% of mass below 1.96
z.ppf(0.975)    # 1.96  — inverse CDF (quantile function)
```

### 5.3 Multivariate Gaussian

Extends the Gaussian to $n$-dimensional vectors:

$$\mathcal{N}(\boldsymbol{\mu}, \Sigma): \quad f(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\!\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)$$

| Term | Shape | Meaning |
|------|-------|---------|
| $\boldsymbol{\mu}$ | $(n,)$ | Mean vector |
| $\Sigma$ | $(n \times n)$ | Covariance matrix — encodes spread and correlations |
| $\Sigma^{-1}$ | $(n \times n)$ | Precision matrix |
| $(\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})$ | scalar | Mahalanobis distance |

```python
mean = np.array([0.0, 0.0])
cov  = np.array([[1.0, 0.8],
                 [0.8, 1.0]])    # correlated: X and Y tend to move together

samples = np.random.multivariate_normal(mean, cov, size=1000)
```

### 5.4 Categorical & Softmax

**Categorical($\mathbf{p}$):** generalization of Bernoulli to $K$ classes.

$$P(X = k) = p_k, \quad \sum_{k=1}^K p_k = 1$$

**Softmax** converts raw scores (logits) $\mathbf{z} \in \mathbb{R}^K$ into a valid probability distribution:

$$\text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$

| Property | Value |
|----------|-------|
| Output range | $(0, 1)$ for each $k$ |
| Output sum | $\sum_k \text{softmax}(\mathbf{z})_k = 1$ |
| Temperature scaling | $\text{softmax}(\mathbf{z}/T)$: $T \to 0$ → one-hot; $T \to \infty$ → uniform |

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.1])
probs  = F.softmax(logits, dim=0)
# tensor([0.659, 0.242, 0.099])
```

### 5.5 Poisson Distribution

**Poisson($\lambda$):** counts the number of events occurring in a fixed interval of time or space, when events happen at a constant average rate $\lambda$ and independently.

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

$$\mathbb{E}[X] = \lambda, \qquad \text{Var}(X) = \lambda$$

> A unique property: mean equals variance. Larger $\lambda$ → more events, more spread.

**Parameter $\lambda$:**

| Symbol | Meaning |
|--------|---------|
| $\lambda$ | Expected number of events per interval (rate) |
| $k$ | Observed count of events |

**Classic applications:**

| Scenario | $\lambda$ example |
|----------|-----------------|
| Number of emails per hour | $\lambda = 3$ |
| Number of photons hitting a sensor | $\lambda = 0.5$ |
| Number of requests to a web server per second | $\lambda = 100$ |
| Number of typos per page | $\lambda = 0.2$ |

**In AI:** Poisson regression models count data (e.g., word frequency, event counts). The Poisson distribution is also related to the exponential distribution: if events arrive at rate $\lambda$, the waiting time between events is $\text{Exp}(\lambda)$.

```python
from scipy import stats
import numpy as np

# λ = 3 events per hour
X = stats.poisson(mu=3)

print(X.pmf(0))   # P(X=0) = e^{-3} ≈ 0.050  — zero events
print(X.pmf(3))   # P(X=3) ≈ 0.224            — exactly λ events
print(X.cdf(5))   # P(X≤5) ≈ 0.916

# Sample counts
counts = np.random.poisson(lam=3, size=1000)
print(counts.mean())   # ≈ 3.0
print(counts.var())    # ≈ 3.0  (mean ≈ variance)
```

**Limit relationship:** Binomial$(n, p)$ with large $n$, small $p$, and $np = \lambda$ converges to Poisson$(\lambda)$.

$$\text{Binom}(n, p) \xrightarrow{n \to \infty,\, np = \lambda} \text{Poisson}(\lambda)$$

### 5.6 Gamma Distribution

**Gamma($\alpha, \beta$):** models the waiting time until $\alpha$ events occur, where events arrive at rate $\beta$.

$$f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, \quad x > 0$$

$$\mathbb{E}[X] = \frac{\alpha}{\beta}, \qquad \text{Var}(X) = \frac{\alpha}{\beta^2}$$

| Parameter | Name | Meaning |
|-----------|------|---------|
| $\alpha > 0$ | Shape | Number of events to wait for; controls shape of distribution |
| $\beta > 0$ | Rate | Event rate; $1/\beta$ is the scale (mean waiting time per event) |
| $\Gamma(\alpha)$ | Gamma function | Generalized factorial: $\Gamma(n) = (n-1)!$ for integers |

**Gamma function:**

$$\Gamma(\alpha) = \int_0^\infty t^{\alpha-1} e^{-t}\,dt, \quad \Gamma(n) = (n-1)!, \quad \Gamma(1/2) = \sqrt{\pi}$$

**Special cases:**

| Parameters | Distribution | Meaning |
|-----------|-------------|---------|
| $\alpha = 1$ | Exponential$(\beta)$ | Waiting time for 1 event |
| $\alpha = k/2$, $\beta = 1/2$ | Chi-squared($k$) | Sum of $k$ squared standard normals |
| $\alpha = n$, $\beta = n/\mu$ | Erlang($n, \mu$) | Integer-shape Gamma |

**In AI:**
- Gamma is the **conjugate prior** for the Poisson rate $\lambda$ and Gaussian precision $1/\sigma^2$
- Chi-squared distribution (a Gamma special case) appears in hypothesis testing
- Bayesian neural networks use Gamma priors on precision parameters

```python
from scipy import stats

# Gamma(alpha=2, beta=1): waiting time for 2 events at rate 1
X = stats.gamma(a=2, scale=1.0)   # scipy uses scale = 1/β

print(X.mean())    # alpha/beta = 2.0
print(X.var())     # alpha/beta^2 = 2.0
print(X.pdf(1.0))  # density at x=1
```

### 5.7 Beta & Dirichlet Distributions

**Beta($\alpha, \beta$):** distribution over probabilities $p \in [0, 1]$.

$$f(p) = \frac{p^{\alpha-1}(1-p)^{\beta-1}}{B(\alpha,\beta)}, \quad p \in [0,1]$$

$$\mathbb{E}[p] = \frac{\alpha}{\alpha+\beta}, \qquad \text{Var}(p) = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$

| Term | Meaning |
|------|---------|
| $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$ | Beta function (normalization constant) |
| $\alpha > 1, \beta > 1$ | Bell-shaped, peaked at $\alpha/(\alpha+\beta)$ |
| $\alpha = \beta = 1$ | Uniform over $[0,1]$ — no preference |
| $\alpha < 1, \beta < 1$ | U-shaped — probability mass at extremes |
| $\alpha + \beta$ | "Concentration" — larger value → tighter distribution |

**Intuition:** $\alpha$ = number of observed successes, $\beta$ = number of observed failures.

| $\alpha$, $\beta$ | Shape | Interpretation |
|---|---|---|
| $1, 1$ | Flat | No prior knowledge |
| $10, 10$ | Peaked at $0.5$ | Strong belief coin is fair |
| $3, 1$ | Skewed right | Believe $p \approx 0.75$ |
| $0.5, 0.5$ | U-shaped | Jeffreys prior — non-informative |

**Beta as conjugate prior for Bernoulli/Binomial:**

$$\text{Prior: } p \sim \text{Beta}(\alpha, \beta)$$
$$\text{Data: } k \text{ successes in } n \text{ trials}$$
$$\text{Posterior: } p \mid \text{data} \sim \text{Beta}(\alpha + k,\; \beta + n - k)$$

The posterior mean is:

$$\mathbb{E}[p \mid \text{data}] = \frac{\alpha + k}{\alpha + \beta + n}$$

> This is a weighted average between the prior mean $\alpha/(\alpha+\beta)$ and the data mean $k/n$.

---

**Dirichlet($\boldsymbol{\alpha}$):** generalization of Beta to $K$-dimensional probability vectors $\mathbf{p} = (p_1, \ldots, p_K)$ on the probability simplex $\sum_k p_k = 1$.

$$f(\mathbf{p}) = \frac{1}{B(\boldsymbol{\alpha})} \prod_{k=1}^K p_k^{\alpha_k - 1}, \quad \mathbf{p} \in \Delta^{K-1}$$

$$\mathbb{E}[p_k] = \frac{\alpha_k}{\sum_j \alpha_j} =: \bar{\alpha}_k, \qquad \text{Var}(p_k) = \frac{\bar{\alpha}_k(1-\bar{\alpha}_k)}{\sum_j \alpha_j + 1}$$

| Parameter | Meaning |
|-----------|---------|
| $\boldsymbol{\alpha} = (\alpha_1, \ldots, \alpha_K)$ | Concentration parameters — pseudo-counts |
| $\alpha_0 = \sum_k \alpha_k$ | Total concentration — higher → sharper distribution |
| $\boldsymbol{\alpha} = (1, \ldots, 1)$ | Uniform on simplex |
| $\boldsymbol{\alpha} = (0.1, \ldots, 0.1)$ | Sparse — concentrated near vertices |

**Dirichlet as conjugate prior for Multinomial:**

$$\text{Prior: } \mathbf{p} \sim \text{Dir}(\boldsymbol{\alpha})$$
$$\text{Data: } \mathbf{c} = (c_1,\ldots,c_K) \text{ counts from Multinomial}(n, \mathbf{p})$$
$$\text{Posterior: } \mathbf{p} \mid \mathbf{c} \sim \text{Dir}(\boldsymbol{\alpha} + \mathbf{c})$$

**In AI:**

| Application | Role |
|-------------|------|
| Topic models (LDA) | Dirichlet prior over topic mixtures |
| Bayesian language models | Dirichlet-Multinomial over word distributions |
| Reinforcement learning | Beta/Dirichlet priors over action probabilities |

```python
import numpy as np
from scipy import stats

# Beta: prior over coin bias
alpha, beta = 2, 5  # believe coin shows heads ~2/7 of the time
X = stats.beta(alpha, beta)
print(X.mean())    # 2/7 ≈ 0.286
print(X.std())     # spread of uncertainty

# After observing 3 heads, 7 tails
alpha_post = alpha + 3   # = 5
beta_post  = beta  + 7   # = 12
print(alpha_post / (alpha_post + beta_post))  # posterior mean ≈ 0.294

# Dirichlet: prior over 3-class probabilities
alpha_dir = np.array([1.0, 1.0, 1.0])   # uniform
samples = np.random.dirichlet(alpha_dir, size=5)
# Each row sums to 1: [[0.3, 0.5, 0.2], [0.1, 0.6, 0.3], ...]
```

---

## 6. Information Theory

### 6.1 Entropy

Entropy $H$ measures the **average uncertainty** (or information content) of a distribution:

$$H(X) = -\sum_x p(x) \log p(x) = \mathbb{E}[-\log p(X)]$$

| Entropy | Meaning |
|---------|---------|
| $H = 0$ | Completely certain — one outcome has probability 1 |
| $H$ maximum | Completely uncertain — uniform distribution |
| Higher $H$ | More spread out, more surprising, more bits needed to encode |

**Example — fair coin vs biased coin:**

$$H(\text{fair}) = -0.5\log 0.5 - 0.5\log 0.5 = 1 \text{ bit}$$
$$H(\text{biased, } p=0.9) = -0.9\log 0.9 - 0.1\log 0.1 \approx 0.47 \text{ bits}$$

```python
import numpy as np

def entropy(p):
    p = np.array(p)
    return -np.sum(p * np.log(p + 1e-12))

entropy([0.5, 0.5])           # 0.693 (nats) = 1.0 bit
entropy([0.9, 0.1])           # 0.325 nats
entropy([1.0, 0.0])           # 0.0 nats — no uncertainty
entropy([0.25, 0.25, 0.25, 0.25])  # 1.386 nats — maximum for 4 classes
```

### 6.2 Cross-Entropy

Cross-entropy measures the average number of bits needed to encode samples from $p$ using a code designed for $q$:

$$H(p, q) = -\sum_x p(x) \log q(x) = \mathbb{E}_p[-\log q(X)]$$

| Term | Meaning |
|------|---------|
| $p$ | True distribution (ground truth labels) |
| $q$ | Predicted distribution (model output) |
| $H(p,q) = H(p)$ | Perfect prediction — $q = p$ |
| $H(p,q) > H(p)$ | Always — model is imperfect |

**Cross-entropy loss in classification:**

$$\mathcal{L} = -\sum_{k=1}^K y_k \log \hat{p}_k$$

For a one-hot label (true class $c$), this simplifies to:

$$\mathcal{L} = -\log \hat{p}_c$$

> Minimizing cross-entropy = maximizing log-likelihood of the true class.

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([[2.0, 1.0, 0.1]])   # model output (unnormalized)
target = torch.tensor([0])                  # true class = 0

loss = F.cross_entropy(logits, target)     # = -log(softmax(2.0)) ≈ 0.407
```

### 6.3 KL Divergence & Total Variation Distance

**KL Divergence** measures how different distribution $q$ is from the true distribution $p$:

$$D_{\text{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_p\!\left[\log \frac{p(X)}{q(X)}\right]$$

**Relationship to cross-entropy:**

$$D_{\text{KL}}(p \| q) = H(p, q) - H(p)$$

| Property | Value |
|----------|-------|
| $D_{\text{KL}}(p \| q) \geq 0$ | Always — zero only when $p = q$ (Gibbs' inequality) |
| $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$ | **Not symmetric** — not a true distance |
| $D_{\text{KL}}(p \| q) = \infty$ | If $q(x) = 0$ anywhere that $p(x) > 0$ |

**Forward vs Reverse KL:**

| | $D_{\text{KL}}(p \| q)$ | $D_{\text{KL}}(q \| p)$ |
|---|---|---|
| Name | Forward KL | Reverse KL |
| Forces $q$ to... | Cover all of $p$ (zero-avoiding) | Fit one mode of $p$ (zero-forcing) |
| Behavior | **Mean-seeking** | **Mode-seeking** |
| Used in | VAE (ELBO), diffusion | Variational inference, RL |

**KL between two Gaussians** (closed form — used in VAEs):

$$D_{\text{KL}}\!\left(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)\right) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

---

**Total Variation Distance (TVD)** is a true metric (symmetric, satisfies triangle inequality):

$$\text{TV}(p, q) = \frac{1}{2}\sum_x |p(x) - q(x)| = \frac{1}{2}\|p - q\|_1$$

| Property | Value |
|----------|-------|
| Range | $[0, 1]$ |
| $\text{TV}(p, q) = 0$ | $p = q$ identically |
| $\text{TV}(p, q) = 1$ | $p$ and $q$ have completely disjoint support |
| Symmetry | $\text{TV}(p, q) = \text{TV}(q, p)$ ✓ |

**Geometric interpretation:** TV is half the $L_1$ distance between distributions. It measures the maximum probability that you can distinguish whether a sample came from $p$ or $q$.

$$\text{TV}(p, q) = \max_{A \subseteq \Omega} |P(A) - Q(A)|$$

**Comparison of divergences:**

| Measure | Symmetric | Range | Meaning |
|---------|-----------|-------|---------|
| KL divergence | No | $[0, \infty)$ | Information gain; used in optimization |
| Total Variation | Yes | $[0, 1]$ | Distinguishability |
| Jensen-Shannon divergence | Yes | $[0, \log 2]$ | Symmetric version of KL |

**Pinsker's inequality** connects KL and TV:

$$\text{TV}(p, q) \leq \sqrt{\frac{1}{2} D_{\text{KL}}(p \| q)}$$

```python
import numpy as np

def tv_distance(p, q):
    p, q = np.array(p), np.array(q)
    return 0.5 * np.sum(np.abs(p - q))

def kl_divergence(p, q):
    p, q = np.array(p) + 1e-12, np.array(q) + 1e-12
    return np.sum(p * np.log(p / q))

p = [0.5, 0.3, 0.2]
q = [0.4, 0.4, 0.2]

print(tv_distance(p, q))    # 0.1
print(kl_divergence(p, q))  # ~0.038

# VAE regularization: KL(N(μ,σ²) || N(0,1))
import torch
def kl_to_standard_normal(mu, log_var):
    return -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp())
```

### 6.4 Mutual Information

Mutual information measures how much knowing $X$ reduces uncertainty about $Y$:

$$I(X; Y) = D_{\text{KL}}(p(X,Y) \| p(X)p(Y)) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X)$$

| Value | Meaning |
|-------|---------|
| $I(X;Y) = 0$ | $X$ and $Y$ are independent — knowing one tells nothing about the other |
| $I(X;Y) = H(X)$ | $Y$ completely determines $X$ |
| Higher $I(X;Y)$ | Stronger dependence between $X$ and $Y$ |

**In AI:** mutual information is used in representation learning, feature selection, and information bottleneck theory to measure how much a representation retains about the input.

---

## 7. Estimation

### 7.1 Bias — Definition

The **bias** of an estimator $\hat{\theta}$ for a true parameter $\theta$ is the systematic error in expectation:

$$\text{Bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta$$

| Value | Meaning |
|-------|---------|
| $\text{Bias}(\hat{\theta}) = 0$ | **Unbiased** — correct on average |
| $\text{Bias}(\hat{\theta}) > 0$ | Overestimates $\theta$ on average |
| $\text{Bias}(\hat{\theta}) < 0$ | Underestimates $\theta$ on average |

**Key insight:** bias is about the *average* over many repeated experiments, not any single estimate.

> Even if $\hat{\theta}$ is wrong for a specific dataset, it is unbiased if $\mathbb{E}[\hat{\theta}] = \theta$ when averaged over all possible datasets of that size.

**Concrete example:** estimating a fair coin's bias $\theta = 0.5$.

- Flip once: $\hat{\theta} = X \in \{0, 1\}$ — always wrong for any single flip, but $\mathbb{E}[\hat{\theta}] = 0.5$ → unbiased.
- Always predict $\hat{\theta} = 0.4$ — wrong on average → biased.

### 7.2 Biased vs Unbiased Estimation

| | Unbiased | Biased |
|---|---|---|
| Definition | $\mathbb{E}[\hat{\theta}] = \theta$ | $\mathbb{E}[\hat{\theta}] \neq \theta$ |
| Error type | Random error only | Systematic + random error |
| Example | Sample mean $\bar{X}$ | MLE variance $\hat{\sigma}^2_\text{MLE}$ |
| Always preferred? | No — can have high variance | Sometimes lower MSE |

**Sample mean is unbiased:**

$$\mathbb{E}[\bar{X}] = \mathbb{E}\!\left[\frac{1}{N}\sum_i X_i\right] = \frac{1}{N}\sum_i \mathbb{E}[X_i] = \mu \checkmark$$

**MLE variance estimator is biased:**

$$\hat{\sigma}^2_\text{MLE} = \frac{1}{N}\sum_i (X_i - \bar{X})^2 \quad \Rightarrow \quad \mathbb{E}[\hat{\sigma}^2_\text{MLE}] = \frac{N-1}{N}\sigma^2 < \sigma^2$$

The MLE systematically underestimates the true variance. Fix: divide by $N-1$ instead.

### 7.3 Sample Variance & the N−1 Correction

**Sample variance** (unbiased estimator of population variance):

$$S^2 = \frac{1}{N-1}\sum_{i=1}^N (X_i - \bar{X})^2$$

**Why $N-1$, not $N$?**

We are estimating $\bar{X}$ from the same data. Using $\bar{X}$ instead of the true $\mu$ introduces a constraint: the deviations $(X_i - \bar{X})$ must sum to zero by definition.

$$\sum_{i=1}^N (X_i - \bar{X}) = 0 \quad \text{(always true)}$$

This constraint means we have only $N-1$ **degrees of freedom**, not $N$ — the last deviation is determined by the others.

**Proof that $S^2$ is unbiased:**

$$\mathbb{E}[S^2] = \mathbb{E}\!\left[\frac{1}{N-1}\sum_i(X_i-\bar{X})^2\right]$$

Expand using $\text{Var}(X_i - \bar{X})$:

$$= \frac{1}{N-1}\,\mathbb{E}\!\left[\sum_i(X_i-\bar{X})^2\right] = \frac{1}{N-1}\cdot(N-1)\sigma^2 = \sigma^2 \checkmark$$

**Intuition:** $\bar{X}$ is always "pulled toward" the data, so the deviations $(X_i - \bar{X})$ are systematically smaller than $(X_i - \mu)$. Dividing by $N-1$ corrects this shrinkage.

**Small sample example:**

Data: $\{2, 4, 6\}$, true $\mu = 4$, true $\sigma^2 = ?$

$$\bar{X} = 4, \quad \hat{\sigma}^2_\text{MLE} = \frac{(2-4)^2+(4-4)^2+(6-4)^2}{3} = \frac{8}{3} \approx 2.67$$

$$S^2 = \frac{8}{2} = 4 \quad \leftarrow \text{unbiased}$$

```python
import numpy as np

data = np.array([2.0, 4.0, 6.0])

var_mle     = np.var(data, ddof=0)    # divide by N   → 2.667 (biased)
var_sample  = np.var(data, ddof=1)    # divide by N-1 → 4.0   (unbiased)

# Verify: repeat 100,000 times and check the average
true_variance = 4.0  # Var for data coming from N(4, σ²=4)
estimates_mle = []
estimates_s2  = []
for _ in range(100_000):
    x = np.random.normal(4.0, 2.0, 3)   # σ² = 4
    estimates_mle.append(np.var(x, ddof=0))
    estimates_s2.append(np.var(x, ddof=1))

print(f"E[MLE var] = {np.mean(estimates_mle):.3f}")   # ≈ 2.667 (biased low)
print(f"E[S²]      = {np.mean(estimates_s2):.3f}")    # ≈ 4.000 (unbiased)
```

### 7.4 Is Unbiased Always Better?

**No.** Unbiased estimators can have very high variance, leading to worse performance in practice.

The mean squared error (MSE) of an estimator decomposes as:

$$\text{MSE}(\hat{\theta}) = \text{Var}(\hat{\theta}) + \text{Bias}(\hat{\theta})^2$$

A biased estimator with low variance can have lower MSE than an unbiased estimator with high variance.

**Example — James-Stein shrinkage estimator:**

For estimating a $d$-dimensional mean vector $\boldsymbol{\mu}$ from $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, I)$:

- Sample mean $\bar{\mathbf{X}}$ is unbiased but has high variance when $d \geq 3$.
- Shrinkage estimator $\hat{\boldsymbol{\mu}}_\text{JS} = \left(1 - \frac{d-2}{\|\mathbf{X}\|^2}\right)\mathbf{X}$ is **biased** but has strictly lower MSE.

**In deep learning:** L2 regularization (weight decay) shrinks weights toward zero — biasing the estimator but reducing overfitting (variance). Lower MSE in practice despite the bias.

| Scenario | Prefer Unbiased? |
|----------|-----------------|
| Small dataset, many parameters | No — high variance overwhelms |
| Large dataset | Yes — bias matters more |
| Safety-critical | Yes — systematic error is dangerous |
| MLE variance with small $N$ | No — use $S^2$ |

### 7.5 Bias, Variance, MSE — and their Relationship

For a predictor $\hat{f}$ trying to approximate the true function $f$:

$$\text{MSE}(\hat{f}(x)) = \mathbb{E}\!\left[(f(x) - \hat{f}(x))^2\right]$$

**Bias-Variance decomposition:**

$$\text{MSE}(\hat{f}(x)) = \underbrace{\left(\mathbb{E}[\hat{f}(x)] - f(x)\right)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\!\left[\left(\hat{f}(x) - \mathbb{E}[\hat{f}(x)]\right)^2\right]}_{\text{Variance}} + \underbrace{\sigma^2_\epsilon}_{\text{Irreducible Noise}}$$

| Component | Source | Fix |
|-----------|--------|-----|
| **Bias²** | Model too simple, wrong assumptions | Larger model, more features |
| **Variance** | Model too complex, sensitive to noise | More data, regularization, dropout |
| **Noise $\sigma^2_\epsilon$** | Inherent randomness in labels | Cannot reduce |

**Tradeoff visualization:**

```
Total Error
    │          Model Complexity →
    │
    │  ↑ Bias (underfitting)
    │      \
    │       \         / ↑ Variance (overfitting)
    │        \_______/
    │         ↑
    │      Sweet spot (min MSE)
    └──────────────────────────
```

**In terms of estimators:**

$$\text{MSE}(\hat{\theta}) = \text{Var}(\hat{\theta}) + \text{Bias}(\hat{\theta})^2$$

**Example:** comparing two estimators of Gaussian variance:

| Estimator | Bias | Variance | MSE |
|-----------|------|----------|-----|
| $S^2 = \frac{1}{N-1}\sum(X_i-\bar{X})^2$ | 0 | $\frac{2\sigma^4}{N-1}$ | $\frac{2\sigma^4}{N-1}$ |
| $\hat{\sigma}^2_\text{MLE} = \frac{1}{N}\sum(X_i-\bar{X})^2$ | $-\sigma^2/N$ | $\frac{2(N-1)\sigma^4}{N^2}$ | slightly lower than $S^2$ |

For large $N$ both are similar; for small $N$, MLE has slightly lower MSE despite being biased!

### 7.6 MLE vs Sample Variance in the Gaussian

Given i.i.d. data $X_1, \ldots, X_N \sim \mathcal{N}(\mu, \sigma^2)$:

**MLE estimates:**

$$\hat{\mu}_\text{MLE} = \bar{X} = \frac{1}{N}\sum_i X_i \quad \text{(unbiased)}$$

$$\hat{\sigma}^2_\text{MLE} = \frac{1}{N}\sum_i (X_i - \bar{X})^2 \quad \text{(biased: underestimates by factor } \frac{N-1}{N}\text{)}$$

**Why MLE variance is biased:**

The MLE uses $\bar{X}$ (estimated from data) as the center. Since $\bar{X}$ is always the closest point to the data, the deviations around $\bar{X}$ are systematically smaller than the true deviations around $\mu$.

$$\mathbb{E}[\hat{\sigma}^2_\text{MLE}] = \frac{N-1}{N}\sigma^2 \neq \sigma^2$$

**Unbiased correction (Bessel's correction):**

$$S^2 = \frac{N}{N-1}\,\hat{\sigma}^2_\text{MLE} = \frac{1}{N-1}\sum_i(X_i-\bar{X})^2$$

$$\mathbb{E}[S^2] = \sigma^2 \checkmark$$

```python
import numpy as np

np.random.seed(42)
N = 5
data = np.random.normal(loc=3.0, scale=2.0, size=N)  # true σ² = 4

mu_mle      = data.mean()
sigma2_mle  = np.var(data, ddof=0)   # divide by N   → biased
sigma2_S2   = np.var(data, ddof=1)   # divide by N-1 → unbiased

print(f"MLE variance:    {sigma2_mle:.3f}")   # likely < 4
print(f"Sample variance: {sigma2_S2:.3f}")    # closer to 4 on average
```

| | Formula | Bias | Use |
|--|---------|------|-----|
| MLE variance | $\frac{1}{N}\sum(x_i-\bar{x})^2$ | $-\sigma^2/N$ | Optimization objectives |
| Sample variance | $\frac{1}{N-1}\sum(x_i-\bar{x})^2$ | 0 | Statistical inference |

### 7.7 Maximum Likelihood Estimation (MLE)

Find parameters $\theta$ that make the observed data $\mathcal{D} = \{x_1, \ldots, x_N\}$ most probable:

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \, p(\mathcal{D} \mid \theta) = \arg\max_\theta \prod_{i=1}^N p(x_i \mid \theta)$$

Taking the log (converts product to sum — more numerically stable):

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^N \log p(x_i \mid \theta)$$

**MLE = minimizing cross-entropy loss** in classification:

$$\mathcal{L}_{\text{CE}} = -\frac{1}{N}\sum_i \log p(y_i \mid x_i, \theta) \quad \Leftrightarrow \quad \hat{\theta}_\text{MLE}$$

**Example — Gaussian MLE:**

Given data $\{x_i\}$, the MLE estimates are:

$$\hat{\mu} = \frac{1}{N}\sum_i x_i, \qquad \hat{\sigma}^2 = \frac{1}{N}\sum_i (x_i - \hat{\mu})^2$$

```python
data = np.array([2.1, 2.5, 3.0, 2.8, 2.3])

mu_mle    = data.mean()         # 2.54
sigma_mle = data.std()          # 0.316 (biased — divides by N)
```

**Properties of MLE:**
- Consistent: $\hat{\theta}_\text{MLE} \to \theta$ as $N \to \infty$
- Asymptotically normal and efficient
- May overfit (no regularization, no prior)

### 7.8 Maximum A Posteriori (MAP)

MAP incorporates a prior belief about $\theta$:

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta \, p(\theta \mid \mathcal{D}) = \arg\max_\theta \left[\log p(\mathcal{D} \mid \theta) + \log p(\theta)\right]$$

| Term | Role |
|------|------|
| $\log p(\mathcal{D} \mid \theta)$ | Log-likelihood — fit to data |
| $\log p(\theta)$ | Log-prior — regularization |

**MAP = MLE + regularization:**

| Prior $p(\theta)$ | Log-prior | Equivalent regularization |
|-------------------|-----------|--------------------------|
| Gaussian $\mathcal{N}(0, \sigma^2)$ | $-\|\theta\|_2^2 / 2\sigma^2$ | L2 (weight decay) |
| Laplace $\text{Lap}(0, b)$ | $-\|\theta\|_1 / b$ | L1 (sparsity) |

> **Key insight:** L2 regularization is the MAP estimate under a Gaussian prior on weights. The regularization strength $\lambda = 1/\sigma^2$ encodes how strongly you believe weights should be near zero.

### 7.9 Conjugate Prior

A **conjugate prior** is a prior distribution that, when combined with a particular likelihood, yields a posterior of the same distributional family.

$$\text{Posterior} \propto \text{Likelihood} \times \text{Prior}$$

If prior and posterior belong to the same family → the prior is **conjugate** to the likelihood.

**Why it matters:** conjugate priors give closed-form posteriors — no numerical integration needed.

**Key conjugate pairs:**

| Likelihood | Conjugate Prior | Posterior |
|-----------|----------------|-----------|
| Bernoulli$(p)$ | Beta$(\alpha, \beta)$ | Beta$(\alpha + k,\; \beta + n - k)$ |
| Binomial$(n, p)$ | Beta$(\alpha, \beta)$ | Beta$(\alpha + k,\; \beta + n - k)$ |
| Multinomial$(n, \mathbf{p})$ | Dirichlet$(\boldsymbol{\alpha})$ | Dirichlet$(\boldsymbol{\alpha} + \mathbf{c})$ |
| Poisson$(\lambda)$ | Gamma$(\alpha, \beta)$ | Gamma$(\alpha + \sum x_i,\; \beta + n)$ |
| Gaussian$(\mu, \sigma^2)$ known $\sigma^2$ | Gaussian$(\mu_0, \sigma_0^2)$ | Gaussian (updated $\mu$ and $\sigma^2$) |
| Gaussian$(\mu, \sigma^2)$ known $\mu$ | Gamma on precision | Gamma (updated) |

**Detailed example — Beta-Bernoulli:**

Prior belief: coin bias $p \sim \text{Beta}(\alpha=2, \beta=5)$ — expect about $2/7 \approx 0.29$.

Observe: 3 heads, 7 tails in 10 flips.

$$\text{Posterior: } p \mid \text{data} \sim \text{Beta}(\underbrace{2+3}_{5},\; \underbrace{5+7}_{12})$$

$$\mathbb{E}[p \mid \text{data}] = \frac{5}{17} \approx 0.29 \quad \leftarrow \text{prior still dominant (weak data)}$$

If instead we observe 30 heads, 70 tails:

$$\text{Posterior} \sim \text{Beta}(32, 75), \quad \mathbb{E}[p \mid \text{data}] = \frac{32}{107} \approx 0.30 \quad \leftarrow \text{data dominates}$$

**Intuition:** $\alpha$ and $\beta$ act as pseudo-counts — as if you had seen $\alpha$ heads and $\beta$ tails before the real experiment. With more data, the posterior is dominated by the data; with little data, the prior dominates.

```python
import numpy as np
from scipy import stats

# Prior: Beta(2, 5)
alpha_prior, beta_prior = 2, 5

# Observe 3H, 7T
k, n = 3, 10
alpha_post = alpha_prior + k            # 5
beta_post  = beta_prior  + (n - k)     # 12

prior     = stats.beta(alpha_prior, beta_prior)
posterior = stats.beta(alpha_post, beta_post)

print(f"Prior mean:     {prior.mean():.3f}")      # 0.286
print(f"Posterior mean: {posterior.mean():.3f}")  # 0.294 (pulled by data)
print(f"Posterior std:  {posterior.std():.3f}")   # uncertainty
```

---

## 8. Variation & Model Fit

### 8.1 Total, Explained, Unexplained Variation

When fitting a regression model $\hat{y} = f(x)$ to data $\{(x_i, y_i)\}$:

Let $\bar{y} = \frac{1}{N}\sum_i y_i$ be the mean of the observed values.

**Total Variation (SST — Sum of Squares Total):**

$$\text{SST} = \sum_{i=1}^N (y_i - \bar{y})^2$$

How spread out are the $y$ values around their mean? This is the variance in $y$ before fitting any model.

**Explained Variation (SSR — Sum of Squares Regression):**

$$\text{SSR} = \sum_{i=1}^N (\hat{y}_i - \bar{y})^2$$

How much of the spread in $y$ is captured by the model predictions $\hat{y}_i$?

**Unexplained Variation (SSE — Sum of Squares Error / Residuals):**

$$\text{SSE} = \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

How much of $y$'s spread remains in the residuals — the part the model failed to explain?

**Fundamental identity:**

$$\text{SST} = \text{SSR} + \text{SSE}$$

| Type | Formula | Meaning |
|------|---------|---------|
| SST (Total) | $\sum(y_i - \bar{y})^2$ | Total variance in $y$ |
| SSR (Explained) | $\sum(\hat{y}_i - \bar{y})^2$ | Variance explained by the model |
| SSE (Residual) | $\sum(y_i - \hat{y}_i)^2$ | Variance not explained — residual error |

**Numerical example:**

Observations: $y = [3, 5, 4, 6, 2]$, model predictions: $\hat{y} = [3.2, 4.8, 4.1, 5.7, 2.2]$

$$\bar{y} = 4.0$$

$$\text{SST} = (3-4)^2 + (5-4)^2 + (4-4)^2 + (6-4)^2 + (2-4)^2 = 10$$

$$\text{SSE} = (3-3.2)^2 + (5-4.8)^2 + (4-4.1)^2 + (6-5.7)^2 + (2-2.2)^2 = 0.18$$

$$\text{SSR} = \text{SST} - \text{SSE} = 9.82$$

### 8.2 Coefficient of Determination R²

$R^2$ (R-squared) measures the fraction of total variation **explained** by the model:

$$R^2 = \frac{\text{SSR}}{\text{SST}} = 1 - \frac{\text{SSE}}{\text{SST}}$$

| $R^2$ | Interpretation |
|-------|---------------|
| $1.0$ | Perfect fit — model explains all variation |
| $0.9$ | Model explains 90% of the variance |
| $0.5$ | Model explains 50% of the variance |
| $0.0$ | Model no better than predicting $\bar{y}$ always |
| $< 0$ | Model is worse than the mean (overfitting or wrong form) |

**Properties:**
- $R^2 \in [0, 1]$ for linear regression with an intercept term
- $R^2$ can be negative for non-linear models or models without intercept
- Adding more features always increases $R^2$ (train) — use **Adjusted $R^2$** to penalize complexity

**Adjusted R²:**

$$\bar{R}^2 = 1 - \frac{\text{SSE}/(N-p-1)}{\text{SST}/(N-1)}$$

where $p$ is the number of predictors. Penalizes extra parameters that don't improve fit.

```python
import numpy as np
from sklearn.metrics import r2_score

y_true = np.array([3.0, 5.0, 4.0, 6.0, 2.0])
y_pred = np.array([3.2, 4.8, 4.1, 5.7, 2.2])

y_mean = y_true.mean()
ss_tot = np.sum((y_true - y_mean)**2)   # 10.0
ss_res = np.sum((y_true - y_pred)**2)   # 0.18
ss_reg = ss_tot - ss_res                # 9.82

r2 = 1 - ss_res / ss_tot               # 0.982
print(f"R² = {r2:.3f}")

# sklearn shortcut
print(r2_score(y_true, y_pred))        # 0.982
```

**In deep learning:** $R^2$ is less commonly used (it's for regression), but the decomposition underlies the intuition behind loss functions and why we care about explained variance in latent representations.

---

## 9. Hypothesis Testing

Hypothesis testing answers: "Is the pattern I see in the data real, or could it be due to chance?"

**Framework:**

1. **Null hypothesis $H_0$:** the default/boring claim (e.g., "the coin is fair", "the drug has no effect")
2. **Alternative hypothesis $H_1$:** what you want to show (e.g., "the coin is biased", "the drug works")
3. Choose a **test statistic** computed from data
4. Compute the **p-value**
5. Reject $H_0$ if p-value $< \alpha$ (significance level, typically 0.05)

### 9.1 P-value

The **p-value** is the probability of observing a test statistic as extreme as (or more extreme than) the one computed from the data, **assuming $H_0$ is true**.

$$\text{p-value} = P(\text{test statistic} \geq t_\text{observed} \mid H_0)$$

| p-value | Interpretation |
|---------|---------------|
| $< 0.05$ | "Statistically significant" — reject $H_0$ |
| $< 0.01$ | Strongly significant |
| $\geq 0.05$ | Fail to reject $H_0$ (not the same as proving $H_0$ true) |

> **Critical misconception:** the p-value is NOT the probability that $H_0$ is true. It is the probability of the data given $H_0$.

**Example — testing a coin:**

Flip a coin 100 times, observe 62 heads. Is it fair ($\theta = 0.5$)?

$$H_0: \theta = 0.5, \quad H_1: \theta \neq 0.5$$

Test statistic: $Z = \frac{\hat{p} - 0.5}{\sqrt{0.5 \cdot 0.5 / 100}} = \frac{0.62 - 0.5}{0.05} = 2.4$

$$\text{p-value} = 2 \cdot P(Z \geq 2.4) \approx 0.016 < 0.05 \quad \Rightarrow \text{reject } H_0$$

**Type I and Type II errors:**

| | $H_0$ True | $H_0$ False |
|--|------------|-------------|
| Reject $H_0$ | **Type I error** (false positive, rate = $\alpha$) | Correct (power = $1-\beta$) |
| Fail to reject $H_0$ | Correct | **Type II error** (false negative, rate = $\beta$) |

```python
from scipy import stats
import numpy as np

# One-sample z-test: is this coin fair?
n_flips = 100
n_heads = 62
p0 = 0.5  # null hypothesis

p_hat = n_heads / n_flips
se    = np.sqrt(p0 * (1 - p0) / n_flips)
z     = (p_hat - p0) / se

p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # two-tailed
print(f"z = {z:.2f}, p-value = {p_value:.4f}")  # z=2.40, p=0.0164

# Using scipy directly
result = stats.binom_test(n_heads, n_flips, p0)  # exact binomial test
```

### 9.2 Confidence Interval

A **confidence interval (CI)** is a range of values that contains the true parameter with a specified probability (confidence level).

$$\bar{X} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{N}}$$

For 95% CI: $z_{0.025} = 1.96$

$$\text{95% CI: } \left[\bar{X} - 1.96\frac{\sigma}{\sqrt{N}},\; \bar{X} + 1.96\frac{\sigma}{\sqrt{N}}\right]$$

**Critical interpretation:**

> "95% confidence" means: if you repeated the experiment many times and computed a CI each time, **95% of those intervals** would contain the true $\mu$.

It does **not** mean: "there is a 95% probability that $\mu$ is in this particular interval" (once computed, $\mu$ is either in it or not).

**Width of the CI:**

$$\text{Width} = 2 \cdot z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{N}}$$

| Factor | Effect on width |
|--------|----------------|
| More data ($N \uparrow$) | Narrower — more precise |
| Higher confidence (95% → 99%) | Wider — need more room |
| Higher variability ($\sigma \uparrow$) | Wider — data more spread out |

**Example:**

$N = 25$ measurements, $\bar{X} = 5.2$, known $\sigma = 1.0$

$$\text{95% CI} = 5.2 \pm 1.96 \cdot \frac{1.0}{\sqrt{25}} = 5.2 \pm 0.392 = [4.808, 5.592]$$

When $\sigma$ is unknown, use $t$-distribution with $N-1$ degrees of freedom:

$$\bar{X} \pm t_{N-1, \alpha/2} \cdot \frac{S}{\sqrt{N}}$$

```python
import numpy as np
from scipy import stats

data = np.random.normal(loc=5.0, scale=1.0, size=25)
N    = len(data)
mean = data.mean()
se   = data.std(ddof=1) / np.sqrt(N)   # standard error

# 95% CI using t-distribution (σ unknown)
ci = stats.t.interval(0.95, df=N-1, loc=mean, scale=se)
print(f"Mean = {mean:.3f}")
print(f"95% CI = [{ci[0]:.3f}, {ci[1]:.3f}]")
```

**Relationship to hypothesis testing:**

If the null hypothesis value $\theta_0$ falls **outside** the 95% CI, the test rejects $H_0$ at $\alpha = 0.05$.

### 9.3 Likelihood-Ratio Test

The **likelihood-ratio test (LRT)** compares how well two models — a null (restricted) and an alternative (full) — fit the data.

**Test statistic:**

$$\Lambda = -2 \log \frac{\mathcal{L}(\hat{\theta}_0)}{\mathcal{L}(\hat{\theta})} = -2\left[\log \mathcal{L}(\hat{\theta}_0) - \log \mathcal{L}(\hat{\theta})\right]$$

| Term | Meaning |
|------|---------|
| $\mathcal{L}(\hat{\theta}_0)$ | Maximum likelihood under $H_0$ (restricted model) |
| $\mathcal{L}(\hat{\theta})$ | Maximum likelihood under $H_1$ (full model) |
| $\Lambda \geq 0$ | Always — full model always fits at least as well |
| $\Lambda$ large | Data much more probable under $H_1$ → reject $H_0$ |

**Wilks' theorem:** under $H_0$, for large $N$:

$$\Lambda \xrightarrow{d} \chi^2(k)$$

where $k$ = number of extra parameters in the full model (degrees of freedom difference).

**Decision rule:** reject $H_0$ if $\Lambda > \chi^2_{k, \alpha}$ (chi-squared critical value).

**Example — testing whether two Gaussians have equal means:**

- $H_0$: $\mu_1 = \mu_2$ (fit one shared mean)
- $H_1$: $\mu_1 \neq \mu_2$ (fit two separate means)
- $k = 1$ extra parameter (one mean vs two means)

**Example — neural network model selection:**

- Small model: 100 parameters, log-likelihood = −500
- Large model: 110 parameters, log-likelihood = −490

$$\Lambda = -2(-490 - (-500)) = 20$$

$$\chi^2_{10, 0.05} = 18.3 \quad \Rightarrow \quad 20 > 18.3 \quad \Rightarrow \text{larger model significantly better}$$

```python
from scipy import stats
import numpy as np

# Simulate: test if two groups have the same mean
np.random.seed(42)
group1 = np.random.normal(5.0, 1.0, 30)
group2 = np.random.normal(5.5, 1.0, 30)

# Log-likelihood under H0: shared mean
mu0 = np.concatenate([group1, group2]).mean()
sigma = 1.0
ll_null = np.sum(stats.norm.logpdf(group1, mu0, sigma)) + \
          np.sum(stats.norm.logpdf(group2, mu0, sigma))

# Log-likelihood under H1: separate means
mu1 = group1.mean()
mu2 = group2.mean()
ll_alt  = np.sum(stats.norm.logpdf(group1, mu1, sigma)) + \
          np.sum(stats.norm.logpdf(group2, mu2, sigma))

Lambda = -2 * (ll_null - ll_alt)   # test statistic
p_val  = 1 - stats.chi2.cdf(Lambda, df=1)   # df = 1 extra parameter

print(f"Λ = {Lambda:.2f}, p-value = {p_val:.4f}")
# If p < 0.05: groups have significantly different means
```

**Why LRT matters in AI:**
- Model selection: compare deep vs shallow networks formally
- Feature selection: test if adding a feature significantly improves fit
- Hypothesis testing in structured models (HMMs, GMMs, etc.)

---

## 10. Sampling & Monte Carlo

### 10.1 Monte Carlo Estimation

Estimate an expectation by averaging over random samples:

$$\mathbb{E}_{x \sim p}[f(x)] \approx \frac{1}{N} \sum_{i=1}^N f(x_i), \quad x_i \sim p$$

The estimate is **unbiased** (converges to the true value) and its variance shrinks as $1/N$.

**In AI:** the policy gradient is a Monte Carlo estimate — roll out $N$ trajectories, average the gradient signal.

```python
# Estimate E[sin(X)] where X ~ Uniform[0, π]
N = 100_000
x = np.random.uniform(0, np.pi, N)
estimate = np.sin(x).mean()   # ≈ 2/π ≈ 0.6366
```

### 10.2 Importance Sampling

Sample from a convenient distribution $q$ instead of the hard-to-sample $p$:

$$\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\!\left[\frac{p(x)}{q(x)} f(x)\right] \approx \frac{1}{N}\sum_i \frac{p(x_i)}{q(x_i)} f(x_i), \quad x_i \sim q$$

| Term | Meaning |
|------|---------|
| $p(x)/q(x)$ | **Importance weight** — corrects for sampling from the wrong distribution |
| $q$ | **Proposal distribution** — easy to sample from |

**In PPO:** the probability ratio $r_t(\theta) = \pi_\theta / \pi_{\theta_\text{old}}$ is exactly an importance weight — it allows reusing data collected under the old policy to estimate expectations under the new policy.

---

## 11. Key Theorems

### 11.1 Law of Large Numbers

As sample size $N \to \infty$, the sample mean converges to the true mean:

$$\bar{X}_N = \frac{1}{N}\sum_{i=1}^N X_i \xrightarrow{N \to \infty} \mathbb{E}[X]$$

**Weak LLN (convergence in probability):**

$$\forall \epsilon > 0: \quad P\!\left(|\bar{X}_N - \mu| > \epsilon\right) \to 0 \quad \text{as } N \to \infty$$

**Strong LLN (almost sure convergence):**

$$P\!\left(\lim_{N\to\infty} \bar{X}_N = \mu\right) = 1$$

**In AI:** this justifies mini-batch gradient descent — averaging over a batch gives an unbiased estimate of the full-dataset gradient, and larger batches give lower variance estimates.

```python
# Verify LLN empirically
true_mean = 5.0
for N in [10, 100, 1000, 10000]:
    samples = np.random.normal(true_mean, 2.0, N)
    print(f"N={N:5d}  sample mean = {samples.mean():.4f}")
# N=   10  sample mean ≈ 4.75
# N=  100  sample mean ≈ 4.98
# N= 1000  sample mean ≈ 4.999
# N=10000  sample mean ≈ 5.0003
```

### 11.2 Central Limit Theorem

The sum (or mean) of $N$ independent random variables converges to a **Gaussian**, regardless of the original distribution:

$$\sqrt{N}\left(\bar{X}_N - \mu\right) \xrightarrow{d} \mathcal{N}(0, \sigma^2) \quad \text{as } N \to \infty$$

Equivalently: $\bar{X}_N \approx \mathcal{N}\!\left(\mu, \frac{\sigma^2}{N}\right)$ for large $N$.

**Requirements:** variables must be i.i.d. with finite mean $\mu$ and finite variance $\sigma^2$.

**Why Gaussians are everywhere in AI:**
- Weights are initialized from Gaussians because the sum of many random initializations is approximately Gaussian
- Noise in stochastic gradient descent is approximately Gaussian
- Residuals and errors tend to be Gaussian in practice
- Sample means (which appear everywhere in statistics) are Gaussian — enabling z-tests and confidence intervals

**Practical rule of thumb:** CLT kicks in around $N \geq 30$ for moderately non-Gaussian distributions.

```python
# CLT: average of Uniform[0,1] variables → Gaussian
for N in [1, 5, 30]:
    means = [np.random.uniform(0, 1, N).mean() for _ in range(10000)]
    print(f"N={N:2d}  mean={np.mean(means):.3f}  std={np.std(means):.3f}")
# N= 1  mean=0.500  std=0.289   ← flat uniform
# N= 5  mean=0.500  std=0.130   ← getting bell-shaped
# N=30  mean=0.500  std=0.053   ← clearly Gaussian

# CLT gives SE of the mean:
# std = σ/√N = 0.289/√30 ≈ 0.053 ✓
```

**CLT enables hypothesis testing:** the sample mean $\bar{X}$ is approximately Gaussian for large $N$, so we can compute exact p-values and confidence intervals even when the data distribution is unknown.

---

## 12. AI Applications

### 12.1 Loss Functions as Likelihoods

Every standard loss function is the negative log-likelihood of a specific distribution:

| Loss | Distribution | Setting |
|------|-------------|---------|
| MSE $\|y - \hat{y}\|_2^2$ | Gaussian $p(y\mid x) = \mathcal{N}(\hat{y}, \sigma^2)$ | Regression |
| Cross-entropy $-\log \hat{p}_c$ | Categorical $p(y\mid x) = \text{Cat}(\hat{\mathbf{p}})$ | Classification |
| BCE $-y\log\hat{p} - (1-y)\log(1-\hat{p})$ | Bernoulli $p(y\mid x) = \text{Bern}(\hat{p})$ | Binary classification |
| MAE $\|y - \hat{y}\|_1$ | Laplace $p(y\mid x) = \text{Lap}(\hat{y}, b)$ | Robust regression |
| Huber loss | Mixture of Gaussian and Laplace | Robust regression |

**Derivation of MSE from MLE:**

$$\log p(y \mid x, \theta) = -\frac{(y - \hat{y})^2}{2\sigma^2} - \text{const}$$

$$\hat{\theta}_\text{MLE} = \arg\max_\theta \sum_i \log p(y_i \mid x_i, \theta) = \arg\min_\theta \sum_i (y_i - \hat{y}_i)^2$$

### 12.2 Bayesian Deep Learning

Standard neural networks output a point estimate $\hat{y}$. Bayesian neural networks output a **distribution** $p(y \mid x, \mathcal{D})$, capturing uncertainty.

**Predictive distribution:**

$$p(y \mid x, \mathcal{D}) = \int p(y \mid x, \theta)\, p(\theta \mid \mathcal{D})\, d\theta$$

This integral is intractable for deep networks — approximate methods are used:

| Method | Idea |
|--------|------|
| **Dropout** at inference | Approximate posterior by averaging over random masks |
| **Deep Ensembles** | Train $K$ networks; average predictions |
| **Variational Inference** | Approximate $p(\theta\mid\mathcal{D})$ with a Gaussian $q_\phi(\theta)$ |

```python
# MC Dropout — uncertainty estimation
model.train()   # keep dropout active during inference
predictions = torch.stack([model(x) for _ in range(50)])

mean = predictions.mean(0)     # expected prediction
var  = predictions.var(0)      # epistemic uncertainty
```

### 12.3 Variational Autoencoders (VAE)

VAEs learn a latent representation $\mathbf{z}$ by maximizing the **Evidence Lower Bound (ELBO)**:

$$\log p(\mathbf{x}) \geq \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\!\left[\log p_\theta(\mathbf{x}\mid\mathbf{z})\right]}_{\text{reconstruction}} - \underbrace{D_{\text{KL}}\!\left(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z})\right)}_{\text{regularization}}$$

| Term | Role |
|------|------|
| $q_\phi(\mathbf{z}\mid\mathbf{x})$ | **Encoder** — maps input to a Gaussian distribution over $\mathbf{z}$ |
| $p_\theta(\mathbf{x}\mid\mathbf{z})$ | **Decoder** — reconstructs input from $\mathbf{z}$ |
| $p(\mathbf{z}) = \mathcal{N}(0, I)$ | **Prior** — standard Gaussian latent space |
| Reconstruction term | Pushes decoder to reproduce $\mathbf{x}$ accurately |
| KL term | Pushes encoder distribution toward the prior — regularizes latent space |

**Reparameterization trick** — makes sampling differentiable:

$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(0, I)$$

```python
import torch
import torch.nn.functional as F

class VAE(torch.nn.Module):
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)       # ε ~ N(0,I)
        return mu + eps * std             # z ~ N(μ, σ²)

    def elbo_loss(self, x, x_recon, mu, log_var):
        recon = F.mse_loss(x_recon, x, reduction='sum')
        kl    = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp())
        return recon + kl
```

### 12.4 Diffusion Models

Diffusion models (DDPM, Stable Diffusion) learn to reverse a gradual noising process.

**Forward process** — add Gaussian noise over $T$ steps until the data becomes pure noise:

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t;\, \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\, \beta_t I)$$

**Closed-form sample at step $t$** (skip $t$ steps at once):

$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t;\, \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\, (1-\bar{\alpha}_t) I)$$

$$\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$$

**Reverse process** — the neural network $\epsilon_\theta$ learns to predict the noise:

$$\mathcal{L}_\text{simple} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\!\left[\|\boldsymbol{\epsilon} - \epsilon_\theta(\mathbf{x}_t, t)\|_2^2\right], \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$$

| Term | Meaning |
|------|---------|
| $\beta_t$ | Noise schedule — how much noise to add at step $t$ |
| $\bar{\alpha}_t$ | Cumulative signal retention — near 1 at $t=0$, near 0 at $t=T$ |
| $\boldsymbol{\epsilon}$ | True noise added to $\mathbf{x}_0$ |
| $\epsilon_\theta(\mathbf{x}_t, t)$ | Network's predicted noise — trained to match $\boldsymbol{\epsilon}$ |

```python
import torch
import torch.nn.functional as F

def forward_diffusion(x0, t, alphas_cumprod):
    noise  = torch.randn_like(x0)
    alpha_t = alphas_cumprod[t].sqrt()
    sigma_t = (1 - alphas_cumprod[t]).sqrt()
    xt     = alpha_t * x0 + sigma_t * noise   # noisy image at step t
    return xt, noise

def diffusion_loss(model, x0, alphas_cumprod, T=1000):
    t   = torch.randint(0, T, (x0.shape[0],))
    xt, noise = forward_diffusion(x0, t, alphas_cumprod)
    predicted_noise = model(xt, t)
    return F.mse_loss(predicted_noise, noise)
```
