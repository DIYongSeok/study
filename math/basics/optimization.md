# Optimization for AI & Robotics

Optimization is the act of choosing the best point from a set of candidates. Training a neural network is minimizing a loss over parameters; fitting an SVM is a constrained quadratic program; planning a robot trajectory is minimizing cost subject to dynamics. Almost every "learning" or "planning" step is an optimization problem, and the *structure* of that problem — convex or not, smooth or not, constrained or not — decides which algorithm works.

This document is the **algorithms-and-theory** reference. The underlying calculus (gradients, Hessians, Taylor expansion) lives in [multivariable_calculus.md](multivariable_calculus.md); the matrix tools (PD matrices, eigendecomposition, least squares) live in [linear_algebra.md](linear_algebra.md).

## Table of Contents

1. [The Optimization Problem](#1-the-optimization-problem)
   - 1.1 [Standard Form & Vocabulary](#11-standard-form--vocabulary)
   - 1.2 [Local vs Global Optima](#12-local-vs-global-optima)
   - 1.3 [Taxonomy of Problems](#13-taxonomy-of-problems)
2. [Convexity](#2-convexity)
   - 2.1 [Convex Sets](#21-convex-sets)
   - 2.2 [Convex Functions](#22-convex-functions)
   - 2.3 [Why Convexity Matters — Local = Global](#23-why-convexity-matters--local--global)
   - 2.4 [Recognizing Convex Functions](#24-recognizing-convex-functions)
   - 2.5 [Strong Convexity & Smoothness](#25-strong-convexity--smoothness)
3. [Optimality Conditions](#3-optimality-conditions)
   - 3.1 [Unconstrained — First & Second Order](#31-unconstrained--first--second-order)
   - 3.2 [Equality Constraints — Lagrange Multipliers](#32-equality-constraints--lagrange-multipliers)
   - 3.3 [Inequality Constraints — KKT](#33-inequality-constraints--kkt)
   - 3.4 [Lagrangian Duality](#34-lagrangian-duality)
4. [Gradient Descent](#4-gradient-descent)
   - 4.1 [The Algorithm](#41-the-algorithm)
   - 4.2 [Step Size & the Descent Lemma](#42-step-size--the-descent-lemma)
   - 4.3 [Convergence Rates](#43-convergence-rates)
   - 4.4 [Conditioning & Zig-Zag](#44-conditioning--zig-zag)
   - 4.5 [Line Search](#45-line-search)
5. [Momentum & Acceleration](#5-momentum--acceleration)
   - 5.1 [Heavy-Ball Momentum](#51-heavy-ball-momentum)
   - 5.2 [Nesterov Accelerated Gradient](#52-nesterov-accelerated-gradient)
6. [Stochastic Optimization](#6-stochastic-optimization)
   - 6.1 [Stochastic Gradient Descent (SGD)](#61-stochastic-gradient-descent-sgd)
   - 6.2 [Mini-Batches & Variance](#62-mini-batches--variance)
   - 6.3 [Learning Rate Schedules](#63-learning-rate-schedules)
7. [Adaptive Methods](#7-adaptive-methods)
   - 7.1 [AdaGrad](#71-adagrad)
   - 7.2 [RMSProp](#72-rmsprop)
   - 7.3 [Adam & AdamW](#73-adam--adamw)
   - 7.4 [Comparison Table](#74-comparison-table)
8. [Second-Order Methods](#8-second-order-methods)
   - 8.1 [Newton's Method](#81-newtons-method)
   - 8.2 [Quasi-Newton — BFGS & L-BFGS](#82-quasi-newton--bfgs--l-bfgs)
   - 8.3 [Gauss–Newton & Levenberg–Marquardt](#83-gaussnewton--levenbergmarquardt)
   - 8.4 [Natural Gradient](#84-natural-gradient)
9. [Constrained & Non-Smooth Methods](#9-constrained--non-smooth-methods)
   - 9.1 [Projected Gradient Descent](#91-projected-gradient-descent)
   - 9.2 [Proximal Gradient & Soft-Thresholding](#92-proximal-gradient--soft-thresholding)
   - 9.3 [Penalty & Augmented Lagrangian](#93-penalty--augmented-lagrangian)
   - 9.4 [Coordinate Descent](#94-coordinate-descent)
10. [Classic Convex Problem Classes](#10-classic-convex-problem-classes)
    - 10.1 [Linear Programming (LP)](#101-linear-programming-lp)
    - 10.2 [Quadratic Programming (QP)](#102-quadratic-programming-qp)
    - 10.3 [Least Squares as Optimization](#103-least-squares-as-optimization)
11. [Non-Convex Landscapes in Deep Learning](#11-non-convex-landscapes-in-deep-learning)
    - 11.1 [Saddle Points Dominate](#111-saddle-points-dominate)
    - 11.2 [Flat vs Sharp Minima](#112-flat-vs-sharp-minima)
    - 11.3 [Practical Tricks](#113-practical-tricks)
12. [AI & Robotics Applications](#12-ai--robotics-applications)
    - 12.1 [Training a Neural Network — End to End](#121-training-a-neural-network--end-to-end)
    - 12.2 [SVM — Duality & the Kernel Trick](#122-svm--duality--the-kernel-trick)
    - 12.3 [PCA as Constrained Optimization](#123-pca-as-constrained-optimization)
    - 12.4 [Trajectory Optimization & MPC](#124-trajectory-optimization--mpc)
    - 12.5 [Policy Gradient & Trust Regions](#125-policy-gradient--trust-regions)

---

## 1. The Optimization Problem

### 1.1 Standard Form & Vocabulary

$$\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x}) \quad \text{subject to} \quad g_i(\mathbf{x}) \le 0,\; i = 1..m, \qquad h_j(\mathbf{x}) = 0,\; j = 1..p$$

| Term | Meaning |
|------|---------|
| $f$ | **Objective** (cost, loss, energy) — the thing to minimize |
| $\mathbf{x}$ | **Decision variables** — parameters $\boldsymbol\theta$ in ML, joint angles or controls in robotics |
| $g_i \le 0$ | **Inequality constraints** |
| $h_j = 0$ | **Equality constraints** |
| Feasible set $\mathcal{F}$ | All $\mathbf{x}$ satisfying every constraint |
| $\mathbf{x}^\star$ | **Minimizer** (argmin); $f^\star = f(\mathbf{x}^\star)$ is the **optimal value** |
| Unconstrained | $m = p = 0$; $\mathcal{F} = \mathbb{R}^n$ |

Maximization is the same problem: $\max f = -\min(-f)$. Everything below is stated for minimization.

### 1.2 Local vs Global Optima

| Type | Definition |
|------|------------|
| **Global minimum** | $f(\mathbf{x}^\star) \le f(\mathbf{x})$ for **all** feasible $\mathbf{x}$ |
| **Local minimum** | $f(\mathbf{x}^\star) \le f(\mathbf{x})$ for all feasible $\mathbf{x}$ in some ball $\|\mathbf{x} - \mathbf{x}^\star\| < r$ |
| **Strict** local minimum | Same with $<$ for $\mathbf{x} \neq \mathbf{x}^\star$ |
| **Critical / stationary point** | $\nabla f(\mathbf{x}^\star) = \mathbf{0}$ (unconstrained) — a minimum, maximum, or saddle |

Gradient-based methods only ever find **critical points**. Whether a critical point is a global minimum depends on the problem's structure — and that structure is convexity (§2).

### 1.3 Taxonomy of Problems

| Axis | Easy end | Hard end |
|------|----------|----------|
| Convexity | **Convex** — every local min is global, solvable to global optimality | **Non-convex** — many local minima and saddles (deep nets) |
| Smoothness | **Smooth** — gradients exist and are Lipschitz | **Non-smooth** — kinks (L1 norm, ReLU, hinge loss) |
| Constraints | **Unconstrained** | Equality + inequality constraints, integer variables |
| Data access | **Deterministic** — exact $f$, $\nabla f$ | **Stochastic** — only noisy mini-batch estimates |
| Scale | $n$ small enough to form and invert $H$ ($n \lesssim 10^4$) | $n \sim 10^9$ parameters — first-order methods only |

> **Key insight:** the dividing line in optimization is not linear vs nonlinear, it is **convex vs non-convex**. Convex problems of enormous size are routinely solved to global optimality; a tiny non-convex problem can be NP-hard.

---

## 2. Convexity

### 2.1 Convex Sets

A set $C \subseteq \mathbb{R}^n$ is **convex** if the line segment between any two of its points stays inside it:

$$\mathbf{x}, \mathbf{y} \in C,\; \theta \in [0, 1] \quad\Longrightarrow\quad \theta\mathbf{x} + (1 - \theta)\mathbf{y} \in C$$

| Convex | Not convex |
|--------|-----------|
| Half-space $\{\mathbf{x} : \mathbf{a}^\top \mathbf{x} \le b\}$ | Union of two disjoint balls |
| Ball, ellipsoid, box | Annulus (ring) |
| Polyhedron $\{\mathbf{x} : A\mathbf{x} \le \mathbf{b}\}$ | Any set with a "dent" |
| Intersection of convex sets | Union of convex sets (in general) |
| Affine subspace $\{\mathbf{x} : A\mathbf{x} = \mathbf{b}\}$ | Integer lattice $\mathbb{Z}^n$ |

The feasible set of a problem with convex $g_i$ and affine $h_j$ is convex (intersection of convex sublevel sets and affine sets).

### 2.2 Convex Functions

$f: \mathbb{R}^n \to \mathbb{R}$ is **convex** if the chord lies above the graph:

$$f(\theta\mathbf{x} + (1 - \theta)\mathbf{y}) \le \theta f(\mathbf{x}) + (1 - \theta) f(\mathbf{y}) \qquad \forall\, \mathbf{x}, \mathbf{y},\; \theta \in [0, 1]$$

Three equivalent characterizations (for differentiable $f$):

| Order | Condition | Reading |
|-------|-----------|---------|
| 0th | Chord above graph (definition) | No "bumps" |
| 1st | $f(\mathbf{y}) \ge f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x})$ | **Tangent plane is a global under-estimator** |
| 2nd | $\nabla^2 f(\mathbf{x}) \succeq 0$ everywhere | Hessian is PSD — curves up in every direction ([linear_algebra.md](linear_algebra.md) §10) |

**Strictly convex** — strict inequality for $\mathbf{x} \neq \mathbf{y}$; guarantees the minimizer is **unique**.
**Concave** — $-f$ is convex. Log-likelihoods are often concave, which is why MLE is often a convex problem.

### 2.3 Why Convexity Matters — Local = Global

**Theorem.** If $f$ is convex, every local minimum is a global minimum.

**Setup.** Let $\mathbf{x}^\star$ be a local minimum: $f(\mathbf{x}^\star) \le f(\mathbf{x})$ for all $\mathbf{x}$ with $\|\mathbf{x} - \mathbf{x}^\star\| < r$. Suppose, for contradiction, some $\mathbf{y}$ has $f(\mathbf{y}) < f(\mathbf{x}^\star)$.

**Step 1 — walk a tiny step toward $\mathbf{y}$.**
Pick $\theta \in (0, 1)$ small enough that $\mathbf{z} = (1 - \theta)\mathbf{x}^\star + \theta\mathbf{y}$ lies inside the ball: $\|\mathbf{z} - \mathbf{x}^\star\| = \theta\|\mathbf{y} - \mathbf{x}^\star\| < r$.

**Step 2 — convexity bounds $f(\mathbf{z})$.**

$$f(\mathbf{z}) \le (1 - \theta) f(\mathbf{x}^\star) + \theta f(\mathbf{y}) < (1 - \theta) f(\mathbf{x}^\star) + \theta f(\mathbf{x}^\star) = f(\mathbf{x}^\star)$$

**Step 3 — contradiction.**
$\mathbf{z}$ is inside the ball but $f(\mathbf{z}) < f(\mathbf{x}^\star)$, contradicting local minimality.

**Conclusion.** No such $\mathbf{y}$ exists; $\mathbf{x}^\star$ is global. $\blacksquare$

> **Corollary (first-order condition is sufficient):** for convex differentiable $f$, $\nabla f(\mathbf{x}^\star) = \mathbf{0}$ implies $\mathbf{x}^\star$ is a global minimum. Proof: the 1st-order characterization gives $f(\mathbf{y}) \ge f(\mathbf{x}^\star) + \mathbf{0}^\top(\mathbf{y} - \mathbf{x}^\star) = f(\mathbf{x}^\star)$ for every $\mathbf{y}$. This is why gradient descent on a convex loss cannot get "stuck" anywhere but the answer.

### 2.4 Recognizing Convex Functions

| Function | Convex? | Why |
|----------|---------|-----|
| $\mathbf{a}^\top \mathbf{x} + b$ | Yes (and concave) | Affine |
| $\|\mathbf{x}\|_p$, any $p \ge 1$ | Yes | Triangle inequality |
| $\mathbf{x}^\top A \mathbf{x}$ | Yes iff $A \succeq 0$ | Hessian is $2A$ |
| $\|A\mathbf{x} - \mathbf{b}\|_2^2$ | Yes | Hessian $2A^\top A \succeq 0$ |
| $e^{ax}$, $-\log x$, $x \log x$ | Yes | Second derivative $> 0$ |
| $\log \sum_i e^{x_i}$ (log-sum-exp) | Yes | Smooth max — underlies softmax cross-entropy |
| $\max_i f_i(\mathbf{x})$, each $f_i$ convex | Yes | Pointwise max preserves convexity |
| $\sum_i w_i f_i$, $w_i \ge 0$ | Yes | Non-negative weighted sum |
| $f(A\mathbf{x} + \mathbf{b})$, $f$ convex | Yes | Composition with affine map |
| Neural network loss in $\boldsymbol\theta$ | **No** | Composition of nonlinearities breaks convexity |

**Operations that preserve convexity:** non-negative sums, pointwise maximum/supremum, affine pre-composition, partial minimization over a convex set. These rules let you certify convexity of large expressions without computing a Hessian.

### 2.5 Strong Convexity & Smoothness

Two constants control how fast gradient methods converge. Both are statements about Hessian eigenvalues.

| Property | Definition | Hessian form | Meaning |
|----------|------------|--------------|---------|
| **$\mu$-strongly convex** | $f(\mathbf{y}) \ge f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x}) + \frac{\mu}{2}\|\mathbf{y} - \mathbf{x}\|^2$ | $\nabla^2 f \succeq \mu I$ | Curves up **at least** as fast as a parabola with curvature $\mu$ |
| **$L$-smooth** | $\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \le L\|\mathbf{x} - \mathbf{y}\|$ | $\nabla^2 f \preceq L I$ | Curves up **at most** as fast as a parabola with curvature $L$; gradient is $L$-Lipschitz |
| **Condition number** | $\kappa = L / \mu \ge 1$ | $\lambda_{\max} / \lambda_{\min}$ | Ratio of steepest to gentlest curvature |

For a quadratic $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top A \mathbf{x}$ with $A \succ 0$: $\mu = \lambda_{\min}(A)$, $L = \lambda_{\max}(A)$ exactly.

> **Sandwich picture:** $\mu$-strong convexity and $L$-smoothness together trap $f$ between two parabolas touching at $\mathbf{x}$. The tighter the sandwich ($\kappa \to 1$), the more $f$ looks like a round bowl and the faster gradient descent converges (§4.3).

---

## 3. Optimality Conditions

### 3.1 Unconstrained — First & Second Order

| Condition | Statement | Type |
|-----------|-----------|------|
| 1st-order **necessary** | $\nabla f(\mathbf{x}^\star) = \mathbf{0}$ | Every local min satisfies it |
| 2nd-order **necessary** | $\nabla^2 f(\mathbf{x}^\star) \succeq 0$ | Every local min satisfies it |
| 2nd-order **sufficient** | $\nabla f = \mathbf{0}$ and $\nabla^2 f(\mathbf{x}^\star) \succ 0$ | Guarantees a strict local min |
| Convex + 1st-order | $f$ convex and $\nabla f = \mathbf{0}$ | Guarantees a **global** min (§2.3) |

Full classification of critical points via the Hessian: [multivariable_calculus.md](multivariable_calculus.md) §8.2.

### 3.2 Equality Constraints — Lagrange Multipliers

**Problem:** $\min f(\mathbf{x})$ subject to $g(\mathbf{x}) = 0$, with $\nabla g(\mathbf{x}^\star) \neq \mathbf{0}$ (regularity).

**Theorem.** At a constrained local minimum $\mathbf{x}^\star$ there exists $\lambda \in \mathbb{R}$ with

$$\nabla f(\mathbf{x}^\star) = \lambda\, \nabla g(\mathbf{x}^\star)$$

**Proof.**

**Setup.** The constraint surface $S = \{\mathbf{x} : g(\mathbf{x}) = 0\}$ is a smooth $(n-1)$-dimensional surface near $\mathbf{x}^\star$ (this is what $\nabla g \neq \mathbf{0}$ buys). Its **tangent space** at $\mathbf{x}^\star$ is

$$T = \{\mathbf{v} : \nabla g(\mathbf{x}^\star)^\top \mathbf{v} = 0\}$$

because for any smooth curve $\mathbf{x}(t)$ on $S$ with $\mathbf{x}(0) = \mathbf{x}^\star$, differentiating $g(\mathbf{x}(t)) = 0$ at $t = 0$ gives $\nabla g^\top \mathbf{x}'(0) = 0$. Conversely every $\mathbf{v} \in T$ is the velocity of some such curve.

**Step 1 — $f$ is stationary along every curve in $S$.**
Take any curve $\mathbf{x}(t) \subset S$ through $\mathbf{x}^\star$ with velocity $\mathbf{v} = \mathbf{x}'(0) \in T$. Then $\phi(t) = f(\mathbf{x}(t))$ has a local minimum at $t = 0$, so

$$0 = \phi'(0) = \nabla f(\mathbf{x}^\star)^\top \mathbf{x}'(0) = \nabla f(\mathbf{x}^\star)^\top \mathbf{v}$$

**Step 2 — so $\nabla f$ is orthogonal to the entire tangent space.**
Step 1 holds for every $\mathbf{v} \in T$, hence $\nabla f(\mathbf{x}^\star) \in T^\perp$.

**Step 3 — identify $T^\perp$.**
$T$ is the orthogonal complement of the single vector $\nabla g$, i.e. $T = \{\nabla g\}^\perp$. Taking the complement again ([linear_algebra.md](linear_algebra.md) §8.4, $(W^\perp)^\perp = W$):

$$T^\perp = \bigl(\{\nabla g\}^\perp\bigr)^\perp = \operatorname{span}\{\nabla g(\mathbf{x}^\star)\}$$

**Conclusion.** $\nabla f(\mathbf{x}^\star) \in \operatorname{span}\{\nabla g(\mathbf{x}^\star)\}$, i.e. $\nabla f(\mathbf{x}^\star) = \lambda \nabla g(\mathbf{x}^\star)$ for some scalar $\lambda$. $\blacksquare$

> **Geometric reading:** the level set of $f$ through $\mathbf{x}^\star$ is *tangent* to the constraint surface. If the two crossed, you could slide along the constraint and lower $f$.

**The Lagrangian repackages this as an unconstrained stationarity problem.** Define

$$\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) - \lambda\, g(\mathbf{x})$$

Then $\nabla_{\mathbf{x}} \mathcal{L} = \mathbf{0}$ is exactly $\nabla f = \lambda \nabla g$, and $\partial \mathcal{L} / \partial \lambda = 0$ is exactly $g(\mathbf{x}) = 0$. One system, $n + 1$ unknowns, $n + 1$ equations.

**Multiple constraints** $h_1 = \cdots = h_p = 0$ with linearly independent gradients: the tangent space is $T = \{\mathbf{v} : \nabla h_j^\top \mathbf{v} = 0 \;\forall j\}$, its complement is $\operatorname{span}\{\nabla h_1, \ldots, \nabla h_p\}$, and the same argument gives

$$\nabla f(\mathbf{x}^\star) = \sum_{j=1}^p \lambda_j \nabla h_j(\mathbf{x}^\star)$$

**Meaning of $\lambda$ — sensitivity / shadow price.** If the constraint is relaxed to $g(\mathbf{x}) = c$ and $f^\star(c)$ is the resulting optimal value, then $\dfrac{d f^\star}{d c} = \lambda$. A large multiplier means the constraint is "expensive"; a zero multiplier means it does not bind.

Worked example (maximize $xy$ subject to $x + y = 10$): [multivariable_calculus.md](multivariable_calculus.md) §9.1. The linear-least-squares instance (KKT block system): [linear_algebra.md](linear_algebra.md) §13.6.

### 3.3 Inequality Constraints — KKT

**Problem:** $\min f(\mathbf{x})$ subject to $g_i(\mathbf{x}) \le 0$ ($i = 1..m$), $h_j(\mathbf{x}) = 0$ ($j = 1..p$).

$$\mathcal{L}(\mathbf{x}, \boldsymbol\mu, \boldsymbol\lambda) = f(\mathbf{x}) + \sum_{i=1}^m \mu_i\, g_i(\mathbf{x}) + \sum_{j=1}^p \lambda_j\, h_j(\mathbf{x})$$

**Karush–Kuhn–Tucker (KKT) conditions** — necessary at a regular local minimum, and **sufficient for a global minimum when the problem is convex**:

| # | Condition | Statement |
|---|-----------|-----------|
| 1 | Stationarity | $\nabla f(\mathbf{x}^\star) + \sum_i \mu_i \nabla g_i(\mathbf{x}^\star) + \sum_j \lambda_j \nabla h_j(\mathbf{x}^\star) = \mathbf{0}$ |
| 2 | Primal feasibility | $g_i(\mathbf{x}^\star) \le 0,\quad h_j(\mathbf{x}^\star) = 0$ |
| 3 | Dual feasibility | $\mu_i \ge 0$ |
| 4 | Complementary slackness | $\mu_i\, g_i(\mathbf{x}^\star) = 0$ for every $i$ |

**Why $\mu_i \ge 0$ (the sign matters).** An inequality constraint $g_i \le 0$ can only push you *into* the feasible region, never pull you out. At an active constraint ($g_i = 0$), moving in direction $-\nabla g_i$ goes into the interior; if $f$ could decrease that way, $\mathbf{x}^\star$ would not be optimal. So $-\nabla f$ must point *outward*, i.e. $-\nabla f = \sum \mu_i \nabla g_i$ with $\mu_i \ge 0$. For an equality constraint both directions are blocked, so $\lambda_j$ has no sign restriction.

**Why complementary slackness.** Each inequality is either

- **active** (tight): $g_i(\mathbf{x}^\star) = 0$ — it participates in stationarity with $\mu_i \ge 0$, or
- **inactive** (slack): $g_i(\mathbf{x}^\star) < 0$ — it is locally irrelevant, so $\mu_i = 0$.

The product $\mu_i g_i = 0$ encodes "at least one of the two is zero." Solving a KKT system in practice means guessing the active set, solving the resulting equality-constrained problem, and checking the signs.

> **In SVMs** (§12.2) complementary slackness is why only the **support vectors** — points with active margin constraints — have non-zero multipliers and therefore the only ones that define the decision boundary.

### 3.4 Lagrangian Duality

**Dual function:** minimize the Lagrangian over the primal variable for fixed multipliers:

$$d(\boldsymbol\mu, \boldsymbol\lambda) = \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol\mu, \boldsymbol\lambda)$$

**Dual problem:** $\max_{\boldsymbol\mu \ge 0,\, \boldsymbol\lambda} d(\boldsymbol\mu, \boldsymbol\lambda)$.

**Theorem (weak duality).** For any $\boldsymbol\mu \ge \mathbf{0}$ and any $\boldsymbol\lambda$: $d(\boldsymbol\mu, \boldsymbol\lambda) \le f^\star$.

**Proof.**

**Setup.** Let $\tilde{\mathbf{x}}$ be any feasible point: $g_i(\tilde{\mathbf{x}}) \le 0$, $h_j(\tilde{\mathbf{x}}) = 0$.

**Step 1 — the penalty terms are non-positive at a feasible point.**
$\mu_i \ge 0$ and $g_i(\tilde{\mathbf{x}}) \le 0$ give $\mu_i g_i(\tilde{\mathbf{x}}) \le 0$; and $\lambda_j h_j(\tilde{\mathbf{x}}) = 0$. So

$$\mathcal{L}(\tilde{\mathbf{x}}, \boldsymbol\mu, \boldsymbol\lambda) = f(\tilde{\mathbf{x}}) + \underbrace{\sum_i \mu_i g_i(\tilde{\mathbf{x}})}_{\le\, 0} + \underbrace{\sum_j \lambda_j h_j(\tilde{\mathbf{x}})}_{=\, 0} \le f(\tilde{\mathbf{x}})$$

**Step 2 — the infimum over all $\mathbf{x}$ is at most the value at $\tilde{\mathbf{x}}$.**

$$d(\boldsymbol\mu, \boldsymbol\lambda) = \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol\mu, \boldsymbol\lambda) \le \mathcal{L}(\tilde{\mathbf{x}}, \boldsymbol\mu, \boldsymbol\lambda) \le f(\tilde{\mathbf{x}})$$

**Conclusion.** This holds for every feasible $\tilde{\mathbf{x}}$, in particular the optimal one: $d(\boldsymbol\mu, \boldsymbol\lambda) \le f^\star$. $\blacksquare$

| Property | Statement |
|----------|-----------|
| **Weak duality** | $d^\star \le f^\star$ always — the dual is a certified **lower bound** on the primal |
| **Duality gap** | $f^\star - d^\star \ge 0$ |
| **Strong duality** | $d^\star = f^\star$; holds for convex problems satisfying **Slater's condition** (a strictly feasible point exists: $g_i(\mathbf{x}) < 0$ for all $i$) |
| **Dual is always concave** | $d$ is a pointwise infimum of affine functions of $(\boldsymbol\mu, \boldsymbol\lambda)$ — even if the primal is non-convex |

**Why duality is useful in practice:**

- **Certificates:** any dual feasible point bounds how far you are from optimal — a stopping criterion.
- **Fewer variables:** the SVM dual has one variable per *data point* rather than per *feature*, and depends on data only through inner products $\mathbf{x}_i^\top \mathbf{x}_j$ — the door to the **kernel trick** (§12.2).
- **Decomposition:** a coupling constraint moves into the objective as a price, and the problem splits into independent sub-problems (dual decomposition, ADMM).

---

## 4. Gradient Descent

### 4.1 The Algorithm

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k \nabla f(\mathbf{x}_k)$$

$\alpha_k > 0$ is the **step size** (ML: **learning rate**). The negative gradient is the direction of steepest descent ([multivariable_calculus.md](multivariable_calculus.md) §3.2).

**Stopping criteria:** $\|\nabla f(\mathbf{x}_k)\| < \varepsilon$, or $|f(\mathbf{x}_{k+1}) - f(\mathbf{x}_k)| < \varepsilon$, or an iteration / wall-clock budget.

### 4.2 Step Size & the Descent Lemma

The whole question of "how big a step" is answered by $L$-smoothness (§2.5).

**Lemma (descent lemma).** If $f$ is $L$-smooth, then for all $\mathbf{x}, \mathbf{y}$:

$$f(\mathbf{y}) \le f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x}) + \frac{L}{2}\|\mathbf{y} - \mathbf{x}\|^2$$

*(Sketch: integrate $\nabla f$ along the segment from $\mathbf{x}$ to $\mathbf{y}$ and bound the change in gradient by Lipschitz continuity.)* This is the "upper parabola" of the sandwich picture.

**Consequence for gradient descent.**

**Setup.** Put $\mathbf{y} = \mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \nabla f(\mathbf{x}_k)$ and write $\mathbf{g} = \nabla f(\mathbf{x}_k)$.

**Step 1 — substitute into the lemma.**

$$f(\mathbf{x}_{k+1}) \le f(\mathbf{x}_k) + \mathbf{g}^\top(-\alpha\mathbf{g}) + \frac{L}{2}\|\alpha\mathbf{g}\|^2 = f(\mathbf{x}_k) - \alpha\Bigl(1 - \frac{L\alpha}{2}\Bigr)\|\mathbf{g}\|^2$$

**Step 2 — read off the step-size rule.**
The bracket is positive iff $\alpha < 2/L$. So any $0 < \alpha < 2/L$ guarantees $f$ decreases whenever $\mathbf{g} \neq \mathbf{0}$. The bracket is maximized at $\alpha = 1/L$, giving the cleanest guarantee:

$$f(\mathbf{x}_{k+1}) \le f(\mathbf{x}_k) - \frac{1}{2L}\|\nabla f(\mathbf{x}_k)\|^2$$

**Conclusion.** With $\alpha = 1/L$, every step reduces $f$ by at least $\frac{1}{2L}\|\nabla f\|^2$. Summing over $k$ shows $\|\nabla f(\mathbf{x}_k)\| \to 0$ — gradient descent converges to a stationary point even for **non-convex** $f$. $\blacksquare$

| Step size | Behavior |
|-----------|----------|
| $\alpha < 2/L$ | Monotone decrease guaranteed |
| $\alpha = 1/L$ | Textbook safe choice |
| $\alpha > 2/L$ | Can overshoot and **diverge** — loss explodes |
| $\alpha \ll 1/L$ | Converges, but painfully slowly |

In deep learning $L$ is unknown and changes across the landscape, which is why learning rate is tuned empirically and scheduled (§6.3).

### 4.3 Convergence Rates

How many iterations to reach $f(\mathbf{x}_k) - f^\star \le \varepsilon$?

| Assumption on $f$ | Rate | Iterations for $\varepsilon$ | Name |
|-------------------|------|------------------------------|------|
| $L$-smooth, non-convex | $\min_k \|\nabla f(\mathbf{x}_k)\|^2 \le \dfrac{2L(f(\mathbf{x}_0) - f^\star)}{k}$ | $O(1/\varepsilon)$ to a stationary point | Sublinear |
| $L$-smooth + convex | $f(\mathbf{x}_k) - f^\star \le \dfrac{L\|\mathbf{x}_0 - \mathbf{x}^\star\|^2}{2k}$ | $O(1/\varepsilon)$ | Sublinear, $O(1/k)$ |
| $L$-smooth + $\mu$-strongly convex | $\|\mathbf{x}_k - \mathbf{x}^\star\| \le \left(1 - \tfrac{1}{\kappa}\right)^k \|\mathbf{x}_0 - \mathbf{x}^\star\|$ | $O(\kappa \log(1/\varepsilon))$ | **Linear** (geometric) |
| Newton, near optimum (§8.1) | $\|\mathbf{x}_{k+1} - \mathbf{x}^\star\| \le C\|\mathbf{x}_k - \mathbf{x}^\star\|^2$ | $O(\log\log(1/\varepsilon))$ | **Quadratic** |

**Where the strongly convex rate comes from — the quadratic case.**

**Setup.** $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top A \mathbf{x} - \mathbf{b}^\top \mathbf{x}$ with $A \succ 0$, so $\nabla f = A\mathbf{x} - \mathbf{b}$ and $\mathbf{x}^\star = A^{-1}\mathbf{b}$.

**Step 1 — the error obeys a linear recursion.**

$$\mathbf{x}_{k+1} - \mathbf{x}^\star = \mathbf{x}_k - \alpha(A\mathbf{x}_k - \mathbf{b}) - \mathbf{x}^\star = (I - \alpha A)(\mathbf{x}_k - \mathbf{x}^\star)$$

**Step 2 — diagonalize.**
$A = V\Lambda V^\top$ ([linear_algebra.md](linear_algebra.md) §2.5), so in eigen-coordinates $\mathbf{e}_k = V^\top(\mathbf{x}_k - \mathbf{x}^\star)$ each component evolves independently:

$$(\mathbf{e}_{k+1})_i = (1 - \alpha\lambda_i)(\mathbf{e}_k)_i \quad\Longrightarrow\quad |(\mathbf{e}_k)_i| = |1 - \alpha\lambda_i|^k\, |(\mathbf{e}_0)_i|$$

**Step 3 — the slowest component sets the rate.**
Convergence requires $|1 - \alpha\lambda_i| < 1$ for every $i$, i.e. $\alpha < 2/\lambda_{\max} = 2/L$ — the same bound as §4.2. The contraction factor is $\rho(\alpha) = \max_i |1 - \alpha\lambda_i| = \max\{|1 - \alpha\mu|,\; |1 - \alpha L|\}$. Balancing the two:

$$\alpha^\star = \frac{2}{L + \mu}, \qquad \rho^\star = \frac{L - \mu}{L + \mu} = \frac{\kappa - 1}{\kappa + 1}$$

**Conclusion.** $\|\mathbf{x}_k - \mathbf{x}^\star\| \le \left(\frac{\kappa - 1}{\kappa + 1}\right)^k \|\mathbf{x}_0 - \mathbf{x}^\star\|$. For large $\kappa$ this is $\approx (1 - 2/\kappa)^k$, so roughly $\kappa/2 \cdot \log(1/\varepsilon)$ iterations. **Iterations scale linearly with the condition number.** $\blacksquare$

### 4.4 Conditioning & Zig-Zag

**Worked example.** $f(x, y) = \frac{1}{2}(x^2 + 10y^2)$, so $A = \operatorname{diag}(1, 10)$, $\mu = 1$, $L = 10$, $\kappa = 10$. Start at $(10, 1)$ with $\alpha = 0.18$ (just under $2/L = 0.2$):

| $k$ | $x_k$ | $y_k$ | Note |
|-----|-------|-------|------|
| 0 | 10.00 | 1.00 | |
| 1 | 8.20 | −0.80 | $y$ overshoots and flips sign |
| 2 | 6.72 | 0.64 | flips again |
| 3 | 5.51 | −0.51 | |
| 10 | 1.37 | 0.11 | $x$ has only decayed by $0.82^{10}$ |

Along $y$ (curvature 10) the step $\alpha\lambda = 1.8$ overshoots and oscillates with factor $|1 - 1.8| = 0.8$. Along $x$ (curvature 1) the step $\alpha\lambda = 0.18$ crawls with factor $0.82$. The single step size cannot serve both directions: **ill-conditioning forces a step small enough for the steepest direction, which is then too small for the flattest one.** In 2D this looks like zig-zagging down a narrow ravine.

**Fixes, all of which appear below:**

| Fix | Idea | Section |
|-----|------|---------|
| Momentum | Average out the oscillation, accumulate speed along the ravine | §5 |
| Preconditioning | Rescale coordinates so $\kappa \to 1$: solve $P^{-1}A$ instead of $A$ | §7, §8 |
| Newton | Use $H^{-1}$ as the perfect preconditioner | §8.1 |
| Feature normalization | Standardize inputs so the loss Hessian is better conditioned | §11.3 |

### 4.5 Line Search

Instead of a fixed $\alpha$, choose it each step along the direction $\mathbf{d}_k$ (e.g. $-\nabla f$).

| Strategy | Rule | Cost |
|----------|------|------|
| **Exact** | $\alpha_k = \arg\min_\alpha f(\mathbf{x}_k + \alpha\mathbf{d}_k)$ | Closed form only for quadratics: $\alpha = \frac{\mathbf{g}^\top\mathbf{g}}{\mathbf{g}^\top A\mathbf{g}}$ |
| **Backtracking (Armijo)** | Start at $\alpha_0$; shrink $\alpha \leftarrow \beta\alpha$ until $f(\mathbf{x}_k + \alpha\mathbf{d}_k) \le f(\mathbf{x}_k) + c\,\alpha\, \nabla f^\top \mathbf{d}_k$ | A few extra function evaluations |
| **Wolfe conditions** | Armijo + a curvature condition preventing steps that are too short | Standard inside BFGS (§8.2) |

Typical constants: $c = 10^{-4}$, $\beta = 0.5$. Line search is standard in deterministic optimization (scipy, L-BFGS) and essentially absent in stochastic deep learning, where the noisy mini-batch loss makes the Armijo test unreliable.

---

## 5. Momentum & Acceleration

### 5.1 Heavy-Ball Momentum

Keep a running **velocity** and let it carry the iterate, like a ball rolling with inertia:

$$\mathbf{v}_{k+1} = \beta\, \mathbf{v}_k - \alpha \nabla f(\mathbf{x}_k), \qquad \mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{v}_{k+1}$$

$\beta \in [0, 1)$ is the momentum coefficient (typical $0.9$). Unrolling: $\mathbf{v}_{k+1} = -\alpha\sum_{i=0}^{k} \beta^{k-i}\nabla f(\mathbf{x}_i)$ — an **exponentially weighted average of past gradients** with effective window $\approx 1/(1 - \beta)$.

**Why it fixes the ravine.** Along the oscillating direction, consecutive gradients alternate sign and cancel in the average. Along the consistent direction they add up: in the steady state $\mathbf{v} \approx -\frac{\alpha}{1 - \beta}\nabla f$, an **effective step size $10\times$ larger** for $\beta = 0.9$.

**Rate on quadratics.** With optimal $\alpha, \beta$, the contraction factor improves from $\frac{\kappa - 1}{\kappa + 1}$ to

$$\rho = \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}$$

so iterations scale with $\sqrt{\kappa}$ instead of $\kappa$. For $\kappa = 10^4$ that is $100\times$ fewer iterations.

### 5.2 Nesterov Accelerated Gradient

Evaluate the gradient at the **look-ahead** point where momentum is about to take you, not at the current point:

$$\mathbf{v}_{k+1} = \beta\, \mathbf{v}_k - \alpha \nabla f(\mathbf{x}_k + \beta\, \mathbf{v}_k), \qquad \mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{v}_{k+1}$$

The gradient acts as a **correction** to the momentum step rather than an independent push — this is what lets it brake before overshooting.

| Method | Convex, $L$-smooth | Strongly convex |
|--------|--------------------|-----------------|
| Gradient descent | $O(1/k)$ | $\left(1 - \tfrac{1}{\kappa}\right)^k$ |
| Nesterov | $O(1/k^2)$ | $\left(1 - \tfrac{1}{\sqrt{\kappa}}\right)^k$ |

Nesterov's $O(1/k^2)$ is **optimal** among all methods that use only gradients — no first-order method can do better in the worst case. Heavy-ball matches it on quadratics but can fail to converge on general convex functions; Nesterov's variant has the guarantee.

---

## 6. Stochastic Optimization

### 6.1 Stochastic Gradient Descent (SGD)

ML losses are averages over data:

$$f(\boldsymbol\theta) = \frac{1}{N}\sum_{i=1}^N \ell_i(\boldsymbol\theta), \qquad \nabla f = \frac{1}{N}\sum_{i=1}^N \nabla \ell_i$$

A full gradient costs $O(N)$ per step. SGD replaces it with the gradient of **one random sample** (or a mini-batch):

$$\boldsymbol\theta_{k+1} = \boldsymbol\theta_k - \alpha_k \nabla \ell_{i_k}(\boldsymbol\theta_k), \qquad i_k \sim \text{Uniform}\{1..N\}$$

**Key property — unbiased:** $\mathbb{E}_{i}[\nabla \ell_i(\boldsymbol\theta)] = \nabla f(\boldsymbol\theta)$. On average SGD steps in the right direction; each individual step is noisy.

| | Full-batch GD | SGD |
|---|---------------|-----|
| Cost per step | $O(N)$ | $O(1)$ or $O(B)$ |
| Steps to $\varepsilon$ (strongly convex) | $O(\kappa\log(1/\varepsilon))$ | $O(1/\varepsilon)$ with decaying $\alpha_k$ |
| Total cost | $O(N\kappa\log(1/\varepsilon))$ | $O(1/\varepsilon)$ — **independent of $N$** |
| Noise | None | Helps escape saddles / sharp minima (§11) |

> **Key insight:** SGD never converges to $\mathbf{x}^\star$ with a constant step size — it hovers in a noise ball of radius $\propto \sqrt{\alpha}$ around it. Decaying $\alpha_k$ shrinks the ball. The Robbins–Monro conditions $\sum \alpha_k = \infty$, $\sum \alpha_k^2 < \infty$ (e.g. $\alpha_k = c/k$) guarantee convergence in the convex case.

### 6.2 Mini-Batches & Variance

Averaging $B$ samples: $\hat{\mathbf{g}} = \frac{1}{B}\sum_{i \in \mathcal{B}} \nabla \ell_i$.

$$\operatorname{Var}[\hat{\mathbf{g}}] = \frac{\sigma^2}{B} \qquad \text{(for i.i.d. samples; } \sigma^2 = \text{per-sample gradient variance)}$$

| Batch size $B$ | Gradient noise | GPU efficiency | Generalization |
|----------------|----------------|----------------|----------------|
| Small (8–64) | High | Poor (under-utilized) | Often better — noise regularizes |
| Large (1k–100k) | Low | Excellent (parallel) | Can be worse — sharp minima; needs LR scaling |

**Linear scaling rule:** when multiplying $B$ by $c$, multiply $\alpha$ by $c$ (up to a limit) — the same total "distance" per epoch. **Gradient accumulation** simulates a large batch on small memory by summing gradients over several forward passes before one update.

### 6.3 Learning Rate Schedules

| Schedule | Formula | Used for |
|----------|---------|----------|
| Constant | $\alpha_k = \alpha_0$ | Quick experiments |
| Step decay | $\alpha_0 \cdot \gamma^{\lfloor k / s \rfloor}$ | Classic CNN training (ResNet: $\times 0.1$ at epochs 30, 60, 90) |
| $1/t$ decay | $\alpha_0 / (1 + \gamma k)$ | Convex theory (Robbins–Monro) |
| **Cosine annealing** | $\alpha_0 \cdot \frac{1}{2}\left(1 + \cos\frac{\pi k}{K}\right)$ | Modern default; smooth decay to 0 |
| **Warmup** | Linear ramp $0 \to \alpha_0$ over first $k_w$ steps, then another schedule | Transformers — avoids early instability when Adam's statistics are unreliable |
| Cyclic / SGDR | Cosine with periodic restarts | Snapshot ensembles |
| One-cycle | Warmup to a high peak, then anneal below $\alpha_0$ | Fast convergence (super-convergence) |

**Warmup + cosine** is the de facto standard for training transformers.

---

## 7. Adaptive Methods

The idea: give each coordinate its **own** step size, scaled inversely to how large its gradients have been. This is a cheap **diagonal preconditioner** (§4.4) — it flattens the ravine without ever forming a Hessian.

### 7.1 AdaGrad

Accumulate squared gradients per coordinate; divide by their root:

$$\mathbf{s}_k = \mathbf{s}_{k-1} + \mathbf{g}_k \odot \mathbf{g}_k, \qquad \boldsymbol\theta_{k+1} = \boldsymbol\theta_k - \frac{\alpha}{\sqrt{\mathbf{s}_k} + \epsilon} \odot \mathbf{g}_k$$

Coordinates that receive large or frequent gradients get small steps; rare, small-gradient coordinates (sparse features, rare words) get large steps. **Problem:** $\mathbf{s}_k$ only grows, so the effective step size decays to zero and training stalls.

### 7.2 RMSProp

Same idea with an **exponential moving average** instead of a sum — old gradients are forgotten:

$$\mathbf{s}_k = \beta_2\, \mathbf{s}_{k-1} + (1 - \beta_2)\, \mathbf{g}_k \odot \mathbf{g}_k, \qquad \boldsymbol\theta_{k+1} = \boldsymbol\theta_k - \frac{\alpha}{\sqrt{\mathbf{s}_k} + \epsilon} \odot \mathbf{g}_k$$

Typical $\beta_2 = 0.99$. $\sqrt{\mathbf{s}_k}$ estimates the recent RMS gradient magnitude per coordinate; dividing by it normalizes every coordinate's step to roughly $\alpha$.

### 7.3 Adam & AdamW

**Adam = RMSProp + momentum + bias correction.** Two moving averages — first moment (mean) and second moment (uncentered variance):

$$\mathbf{m}_k = \beta_1 \mathbf{m}_{k-1} + (1 - \beta_1)\, \mathbf{g}_k \qquad \text{(momentum)}$$
$$\mathbf{v}_k = \beta_2 \mathbf{v}_{k-1} + (1 - \beta_2)\, \mathbf{g}_k \odot \mathbf{g}_k \qquad \text{(RMS scaling)}$$
$$\hat{\mathbf{m}}_k = \frac{\mathbf{m}_k}{1 - \beta_1^k}, \qquad \hat{\mathbf{v}}_k = \frac{\mathbf{v}_k}{1 - \beta_2^k} \qquad \text{(bias correction)}$$
$$\boldsymbol\theta_{k+1} = \boldsymbol\theta_k - \alpha\, \frac{\hat{\mathbf{m}}_k}{\sqrt{\hat{\mathbf{v}}_k} + \epsilon}$$

Defaults: $\alpha = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

**Why bias correction.** With $\mathbf{m}_0 = \mathbf{0}$, after one step $\mathbf{m}_1 = (1 - \beta_1)\mathbf{g}_1 = 0.1\,\mathbf{g}_1$ — biased toward zero by a factor $1 - \beta_1^k$. Dividing by that factor removes the bias; it matters only for the first $\sim 1/(1 - \beta_2) = 1000$ steps, which is also why warmup helps.

**Reading the update.** Per coordinate, the step is $\alpha \cdot \hat m / \sqrt{\hat v} \approx \alpha \cdot \text{(mean of } g) / \text{(RMS of } g)$ — a **signal-to-noise ratio** bounded roughly by $\alpha$ in magnitude. Adam is therefore insensitive to the *scale* of the gradient, which is why one learning rate works across very different layers.

**AdamW — decoupled weight decay.** L2 regularization $\frac{\lambda}{2}\|\boldsymbol\theta\|^2$ adds $\lambda\boldsymbol\theta$ to the gradient, which Adam then *rescales* by $1/\sqrt{\hat{\mathbf{v}}}$ — so the effective decay differs per coordinate and is weakened for large-gradient weights. AdamW applies decay **directly** to the parameters, outside the adaptive scaling:

$$\boldsymbol\theta_{k+1} = \boldsymbol\theta_k - \alpha\left(\frac{\hat{\mathbf{m}}_k}{\sqrt{\hat{\mathbf{v}}_k} + \epsilon} + \lambda\,\boldsymbol\theta_k\right)$$

AdamW is the default optimizer for transformers and most modern large-scale training.

### 7.4 Comparison Table

| Optimizer | State per parameter | Momentum | Per-coordinate scaling | Typical use |
|-----------|--------------------|----------|-----------------------|-------------|
| SGD | 0 | No | No | Baseline, convex problems |
| SGD + momentum / Nesterov | 1 | Yes | No | CNNs on vision (often best final accuracy) |
| AdaGrad | 1 | No | Cumulative | Sparse features, online learning |
| RMSProp | 1 | No | Moving average | RNNs (historically) |
| Adam | 2 | Yes | Moving average | General default |
| **AdamW** | 2 | Yes | Moving average + decoupled decay | **Transformers, LLMs** |
| L-BFGS | $2m$ vectors | Curvature history | Full (low-rank) | Small deterministic problems, physics-informed nets |

> **Rule of thumb:** AdamW with warmup + cosine schedule is the safe default. SGD + momentum can reach slightly better generalization on vision tasks but needs more tuning. Adaptive methods are essentially mandatory for transformers because per-layer gradient scales differ by orders of magnitude.

---

## 8. Second-Order Methods

### 8.1 Newton's Method

Minimize the local quadratic model ([multivariable_calculus.md](multivariable_calculus.md) §7.3) exactly at each step:

$$f(\mathbf{x}_k + \Delta) \approx f(\mathbf{x}_k) + \nabla f^\top \Delta + \tfrac{1}{2}\Delta^\top H \Delta \quad\Longrightarrow\quad \nabla_\Delta = \mathbf{0} \;\Rightarrow\; \Delta = -H^{-1}\nabla f$$

$$\mathbf{x}_{k+1} = \mathbf{x}_k - H(\mathbf{x}_k)^{-1}\nabla f(\mathbf{x}_k)$$

| Property | Newton | Gradient descent |
|----------|--------|------------------|
| Preconditioning | Perfect — $H^{-1}$ makes every direction curvature 1; **$\kappa$-independent** | None |
| On a quadratic | Exact minimizer in **one** step | $\sim\kappa$ steps |
| Rate near optimum | **Quadratic** — digits of accuracy double per step | Linear |
| Cost per step | $O(n^2)$ memory, $O(n^3)$ solve | $O(n)$ |
| Far from optimum | Can diverge; needs damping $\alpha < 1$ or trust region | Robust |
| Non-convex | Heads toward *any* critical point, including saddles and maxima ($H \not\succ 0$) | Descends |

**Damped Newton:** $\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k H^{-1}\nabla f$ with line search, and regularize $H \leftarrow H + \rho I$ when $H$ is not PD. The $O(n^3)$ cost rules Newton out for deep nets ($n \sim 10^9$) but makes it the method of choice for $n \lesssim 10^4$ — logistic regression, robotics trajectory problems, small MLPs.

### 8.2 Quasi-Newton — BFGS & L-BFGS

Never form $H$; **build an approximation $B_k \approx H$ from gradient differences** as you go. Each step observes the pair

$$\mathbf{s}_k = \mathbf{x}_{k+1} - \mathbf{x}_k, \qquad \mathbf{y}_k = \nabla f(\mathbf{x}_{k+1}) - \nabla f(\mathbf{x}_k)$$

and the **secant condition** $B_{k+1}\mathbf{s}_k = \mathbf{y}_k$ says the approximate Hessian must explain the observed gradient change (a finite-difference curvature measurement along $\mathbf{s}_k$).

**BFGS** picks the update to $B_k$ (in fact directly to $B_k^{-1}$) that satisfies the secant condition, stays symmetric PD, and changes $B_k$ as little as possible — a rank-2 correction:

$$B_{k+1}^{-1} = \left(I - \frac{\mathbf{s}_k\mathbf{y}_k^\top}{\mathbf{y}_k^\top\mathbf{s}_k}\right) B_k^{-1} \left(I - \frac{\mathbf{y}_k\mathbf{s}_k^\top}{\mathbf{y}_k^\top\mathbf{s}_k}\right) + \frac{\mathbf{s}_k\mathbf{s}_k^\top}{\mathbf{y}_k^\top\mathbf{s}_k}$$

Superlinear convergence, $O(n^2)$ per step, no Hessian ever computed.

**L-BFGS** (limited memory) stores only the last $m \approx 5$–$20$ pairs $(\mathbf{s}_i, \mathbf{y}_i)$ and applies $B_k^{-1}\nabla f$ via a two-loop recursion in $O(mn)$. This is the workhorse for **deterministic** large-scale problems: `scipy.optimize.minimize(method="L-BFGS-B")`, physics-informed neural networks, structure-from-motion, and full-batch fine-tuning. It works poorly with mini-batch noise because the secant pairs become inconsistent.

### 8.3 Gauss–Newton & Levenberg–Marquardt

For **nonlinear least squares** $f(\mathbf{x}) = \frac{1}{2}\|\mathbf{r}(\mathbf{x})\|^2$ with residual $\mathbf{r}: \mathbb{R}^n \to \mathbb{R}^m$ and Jacobian $J = \partial\mathbf{r}/\partial\mathbf{x}$:

$$\nabla f = J^\top \mathbf{r}, \qquad H = J^\top J + \sum_i r_i \nabla^2 r_i$$

**Gauss–Newton** drops the second term (small when residuals are small or nearly linear) and uses $H \approx J^\top J$:

$$\Delta = -(J^\top J)^{-1} J^\top \mathbf{r}$$

This is exactly a **linear least-squares problem** at each step ([linear_algebra.md](linear_algebra.md) §13.2: linearize $\mathbf{r}(\mathbf{x} + \Delta) \approx \mathbf{r} + J\Delta$ and solve the normal equations). $J^\top J$ is always PSD — Gauss–Newton never heads for a maximum.

**Levenberg–Marquardt** adds a damping term that interpolates between Gauss–Newton and gradient descent:

$$\Delta = -(J^\top J + \rho I)^{-1} J^\top \mathbf{r}$$

| $\rho$ | Behavior |
|--------|----------|
| $\rho \to 0$ | Gauss–Newton: fast when the model is good |
| $\rho \to \infty$ | Small gradient-descent step $-\frac{1}{\rho}J^\top\mathbf{r}$: safe when it is not |

Adaptive rule: if the step decreased $f$, shrink $\rho$; otherwise increase $\rho$ and retry. LM is the standard solver for **camera calibration, bundle adjustment, SLAM, robot inverse kinematics** ([multivariable_calculus.md](multivariable_calculus.md) §12.3), and curve fitting (`scipy.optimize.least_squares`).

### 8.4 Natural Gradient

When parameters define a **probability distribution** $p_{\boldsymbol\theta}$, Euclidean distance in $\boldsymbol\theta$ is the wrong metric — a small change in $\boldsymbol\theta$ can change $p_{\boldsymbol\theta}$ a lot or not at all. Measure steps by KL divergence instead ([statistics_probability.md](statistics_probability.md) §5.3). Locally, $\text{KL}(p_{\boldsymbol\theta} \,\|\, p_{\boldsymbol\theta + \Delta}) \approx \frac{1}{2}\Delta^\top F \Delta$ where

$$F = \mathbb{E}_{\mathbf{x} \sim p_{\boldsymbol\theta}}\left[\nabla_{\boldsymbol\theta}\log p_{\boldsymbol\theta}(\mathbf{x})\, \nabla_{\boldsymbol\theta}\log p_{\boldsymbol\theta}(\mathbf{x})^\top\right]$$

is the **Fisher information matrix**. The steepest-descent direction under this metric is

$$\Delta = -\alpha\, F^{-1} \nabla f$$

Natural gradient is **invariant to reparameterization** — the same distributional path regardless of how $\boldsymbol\theta$ is coordinatized. $F$ is PSD and equals the expected Hessian of the negative log-likelihood, so this is a Newton-like method that never goes uphill. Used in **TRPO / natural policy gradient** (§12.5) and approximated by **K-FAC** for neural nets; Adam's $\sqrt{\hat{\mathbf{v}}}$ can be read as a crude diagonal $F^{1/2}$.

---

## 9. Constrained & Non-Smooth Methods

### 9.1 Projected Gradient Descent

For $\min f(\mathbf{x})$ over a convex set $C$: take a gradient step, then **project back** onto $C$:

$$\mathbf{x}_{k+1} = \Pi_C\bigl(\mathbf{x}_k - \alpha\nabla f(\mathbf{x}_k)\bigr), \qquad \Pi_C(\mathbf{y}) = \arg\min_{\mathbf{x} \in C}\|\mathbf{x} - \mathbf{y}\|_2$$

Works whenever projection is cheap:

| Set $C$ | Projection $\Pi_C(\mathbf{y})$ |
|---------|--------------------------------|
| Box $[\mathbf{l}, \mathbf{u}]$ | $\operatorname{clip}(\mathbf{y}, \mathbf{l}, \mathbf{u})$ per coordinate |
| L2 ball of radius $r$ | $\mathbf{y} \cdot \min(1, r/\|\mathbf{y}\|)$ |
| Non-negative orthant | $\max(\mathbf{y}, \mathbf{0})$ |
| Probability simplex | Sort-and-threshold in $O(n\log n)$ |
| Affine set $A\mathbf{x} = \mathbf{b}$ | $\mathbf{y} - A^\top(AA^\top)^{-1}(A\mathbf{y} - \mathbf{b})$ ([linear_algebra.md](linear_algebra.md) §9.2) |

Same convergence rates as unconstrained GD. **Gradient clipping** in deep learning is projection of the *gradient* onto an L2 ball — the same operation applied to $\mathbf{g}$ instead of $\mathbf{x}$.

### 9.2 Proximal Gradient & Soft-Thresholding

For composite objectives **smooth + non-smooth**:

$$\min_{\mathbf{x}} f(\mathbf{x}) + h(\mathbf{x}), \qquad f \text{ smooth}, \; h \text{ convex but possibly non-differentiable}$$

Take a gradient step on $f$, then apply the **proximal operator** of $h$ — a "denoising" step that trades closeness to the gradient step against making $h$ small:

$$\mathbf{x}_{k+1} = \operatorname{prox}_{\alpha h}\bigl(\mathbf{x}_k - \alpha\nabla f(\mathbf{x}_k)\bigr), \qquad \operatorname{prox}_{\alpha h}(\mathbf{y}) = \arg\min_{\mathbf{x}} \left\{ h(\mathbf{x}) + \frac{1}{2\alpha}\|\mathbf{x} - \mathbf{y}\|^2 \right\}$$

Projection is the special case $h = $ indicator of $C$. The important new case is the **L1 norm**, $h(\mathbf{x}) = \lambda\|\mathbf{x}\|_1$, whose prox is **soft-thresholding**, applied per coordinate:

$$\operatorname{prox}_{\alpha\lambda\|\cdot\|_1}(y)_i = \operatorname{sign}(y_i)\,\max(|y_i| - \alpha\lambda,\; 0) = \begin{cases} y_i - \alpha\lambda & y_i > \alpha\lambda \\ 0 & |y_i| \le \alpha\lambda \\ y_i + \alpha\lambda & y_i < -\alpha\lambda \end{cases}$$

Any coordinate whose gradient step lands within $\alpha\lambda$ of zero is set to **exactly zero** — this is *why* L1 regularization (Lasso, [linear_algebra.md](linear_algebra.md) §13.5) produces sparse solutions, whereas L2 only shrinks. The resulting algorithm is **ISTA**; with Nesterov acceleration it is **FISTA**.

### 9.3 Penalty & Augmented Lagrangian

Turn constraints into cost terms so an unconstrained solver can be used.

**Quadratic penalty:**

$$\min_{\mathbf{x}} f(\mathbf{x}) + \frac{\rho}{2}\sum_j h_j(\mathbf{x})^2$$

Exact only as $\rho \to \infty$, but large $\rho$ makes the problem ill-conditioned ($\kappa \propto \rho$). This is the "soft constraint" used everywhere in robotics cost functions and in physics-informed losses.

**Augmented Lagrangian (method of multipliers):** add the multiplier term too, and update the multipliers between solves:

$$\mathcal{L}_\rho(\mathbf{x}, \boldsymbol\lambda) = f(\mathbf{x}) + \boldsymbol\lambda^\top \mathbf{h}(\mathbf{x}) + \frac{\rho}{2}\|\mathbf{h}(\mathbf{x})\|^2, \qquad \boldsymbol\lambda \leftarrow \boldsymbol\lambda + \rho\, \mathbf{h}(\mathbf{x}_k)$$

Converges to the exact constrained solution with **finite** $\rho$ — the multiplier absorbs the constraint force so $\rho$ need not blow up. **ADMM** (alternating direction method of multipliers) applies this to problems of the form $f(\mathbf{x}) + g(\mathbf{z})$ s.t. $A\mathbf{x} + B\mathbf{z} = \mathbf{c}$, alternating minimization over $\mathbf{x}$ and $\mathbf{z}$; it is the standard solver for large distributed convex problems and for MPC (§12.4).

**Barrier / interior-point:** replace $g_i \le 0$ with $-\frac{1}{t}\log(-g_i)$ in the objective, which is $+\infty$ outside the feasible region. Increase $t$ and re-solve with Newton. This is what commercial LP/QP solvers do (§10).

### 9.4 Coordinate Descent

Minimize over one coordinate (or block) at a time, cycling or sampling randomly:

$$x_i \leftarrow \arg\min_{x_i} f(x_1, \ldots, x_i, \ldots, x_n)$$

Works when each 1D sub-problem is cheap or closed-form. **Lasso** (`glmnet`, sklearn), **SMO for SVMs** (two coordinates at a time, closed-form), and **Gibbs sampling / mean-field variational inference** are all coordinate methods. Can be much faster than GD when the coordinates are weakly coupled; fails on non-smooth objectives whose kinks are not axis-aligned.

---

## 10. Classic Convex Problem Classes

### 10.1 Linear Programming (LP)

$$\min_{\mathbf{x}} \mathbf{c}^\top \mathbf{x} \quad \text{s.t.} \quad A\mathbf{x} \le \mathbf{b},\; \mathbf{x} \ge \mathbf{0}$$

Linear objective, polyhedral feasible set. The optimum is always at a **vertex** of the polyhedron (linear functions have no interior critical points). **Simplex** walks from vertex to vertex; **interior-point** methods cut through the middle with a barrier (§9.3) in polynomial time. The LP dual is another LP: $\max \mathbf{b}^\top \mathbf{y}$ s.t. $A^\top \mathbf{y} \ge \mathbf{c}$, and strong duality always holds for feasible LPs.

**In AI/robotics:** optimal transport (earth mover's distance, Wasserstein), scheduling, resource allocation, linear-constraint verification of neural nets.

### 10.2 Quadratic Programming (QP)

$$\min_{\mathbf{x}} \tfrac{1}{2}\mathbf{x}^\top P \mathbf{x} + \mathbf{q}^\top \mathbf{x} \quad \text{s.t.} \quad A\mathbf{x} \le \mathbf{b},\; C\mathbf{x} = \mathbf{d}$$

Convex iff $P \succeq 0$ ([linear_algebra.md](linear_algebra.md) §10). Equality-only QPs reduce to one linear solve — the KKT block system ([linear_algebra.md](linear_algebra.md) §13.6). Inequality QPs use active-set methods (small, warm-startable — good for MPC) or interior-point (large). Solvers: OSQP, quadprog, CVXPY as a modeling layer.

**In AI/robotics:** SVM training (§12.2), MPC (§12.4), whole-body control of legged robots (torques as QP variables, contacts as constraints), portfolio optimization.

### 10.3 Least Squares as Optimization

| Problem | Objective | Solution | Reference |
|---------|-----------|----------|-----------|
| Ordinary LS | $\|A\mathbf{x} - \mathbf{b}\|^2$ | $A^\top A\,\mathbf{x} = A^\top \mathbf{b}$ | LA §13.2 |
| Ridge | $\|A\mathbf{x} - \mathbf{b}\|^2 + \lambda\|\mathbf{x}\|^2$ | $(A^\top A + \lambda I)\mathbf{x} = A^\top \mathbf{b}$ | LA §13.5 |
| Lasso | $\|A\mathbf{x} - \mathbf{b}\|^2 + \lambda\|\mathbf{x}\|_1$ | No closed form — ISTA / coordinate descent | §9.2, §9.4 |
| Constrained LS | $\|A\mathbf{x} - \mathbf{b}\|^2$ s.t. $C\mathbf{x} = \mathbf{d}$ | KKT block system | LA §13.6 |
| Nonlinear LS | $\|\mathbf{r}(\mathbf{x})\|^2$ | Gauss–Newton / LM | §8.3 |

Least squares is the one non-trivial optimization problem with a closed-form solution — which is why it appears as the inner loop of so many other methods (Gauss–Newton, Newton on a quadratic model, Kalman filtering).

---

## 11. Non-Convex Landscapes in Deep Learning

### 11.1 Saddle Points Dominate

A neural network loss $\mathcal{L}(\boldsymbol\theta)$ is non-convex: symmetries (permuting hidden units), nonlinear activations, and products of weight matrices all break the chord condition. The classical fear was "getting stuck in a bad local minimum." In high dimensions the picture is different:

| Fact | Why it matters |
|------|----------------|
| At a random critical point in $n$ dimensions, each Hessian eigenvalue is roughly equally likely to be $+$ or $-$ | A local minimum needs **all** $n$ signs positive — probability $\sim 2^{-n}$. Almost every critical point is a **saddle** |
| Local minima that do exist tend to have loss close to the global minimum | "Bad" local minima are rare in over-parameterized nets |
| Saddles have near-zero gradient in many directions | Plain GD **slows to a crawl** near them; it does not stop, but it looks stuck |

**Escaping saddles:** SGD noise (§6.1) and momentum (§5) push out along the negative-curvature directions. This is one reason SGD variants beat Newton-type methods in deep learning — Newton is *attracted* to saddles ($H^{-1}\nabla f$ steps toward any critical point).

### 11.2 Flat vs Sharp Minima

Two minima with equal training loss can generalize very differently:

| | Flat minimum | Sharp minimum |
|---|--------------|---------------|
| Hessian eigenvalues | Small — wide basin | Large — narrow spike |
| Under distribution shift (test set) | Loss barely changes | Loss jumps |
| Reached by | Small batches, high LR, SGD noise | Large batches, low LR |

The SGD noise ball (§6.1) cannot sit inside a basin narrower than itself, so noise **selects** flat minima. **Sharpness-Aware Minimization (SAM)** makes this explicit: minimize $\max_{\|\boldsymbol\epsilon\| \le \rho} \mathcal{L}(\boldsymbol\theta + \boldsymbol\epsilon)$ — the worst loss in a neighborhood — by taking an ascent step then a descent step.

### 11.3 Practical Tricks

| Trick | What it fixes | Mechanism |
|-------|---------------|-----------|
| **Input / feature normalization** | Ill-conditioning (§4.4) | Makes the loss Hessian closer to isotropic |
| **Batch / layer norm** | Ill-conditioning inside the network | Re-normalizes activations every layer so gradient scales stay comparable |
| **Careful initialization** (He, Xavier) | Vanishing / exploding gradients | Keeps activation and gradient variance $\approx 1$ through depth |
| **Residual connections** | Optimization difficulty in deep nets | Gives the gradient an identity path; smooths the landscape |
| **Gradient clipping** | Exploding gradients, rare huge steps | Projection of $\mathbf{g}$ onto an L2 ball (§9.1) |
| **Warmup** | Early instability with adaptive methods | Small LR until Adam's $\hat{\mathbf{v}}$ is a reliable estimate |
| **Weight decay** | Overfitting; also improves conditioning | L2 penalty — a strongly convex term $\frac{\lambda}{2}\|\boldsymbol\theta\|^2$ raises $\mu$ |
| **Mixed precision + loss scaling** | Underflow in fp16 gradients | Multiply loss by $2^k$ before backward, divide after |

> **Key insight:** almost every deep-learning "trick" is a way of **improving the conditioning** of the loss or **controlling the noise** of the gradient estimate. The optimizer itself (AdamW) changes rarely; what evolves is the landscape it is asked to descend.

---

## 12. AI & Robotics Applications

### 12.1 Training a Neural Network — End to End

Putting the pieces together for one training step of a transformer:

| Step | Operation | Section |
|------|-----------|---------|
| 1 | Sample a mini-batch $\mathcal{B}$ of size $B$ | §6.2 |
| 2 | Forward pass → loss $\mathcal{L}_\mathcal{B}(\boldsymbol\theta) = \frac{1}{B}\sum_{i \in \mathcal{B}}\ell_i$ | LA §15.1–15.2 |
| 3 | Backward pass → $\mathbf{g} = \nabla_{\boldsymbol\theta}\mathcal{L}_\mathcal{B}$ (unbiased estimate of full gradient) | MC §4.3, LA §12 |
| 4 | Clip: $\mathbf{g} \leftarrow \mathbf{g}\cdot\min(1, c/\|\mathbf{g}\|)$ | §9.1 |
| 5 | Update Adam moments $\mathbf{m}, \mathbf{v}$; bias-correct | §7.3 |
| 6 | Learning rate from schedule: $\alpha_k$ = warmup then cosine | §6.3 |
| 7 | AdamW step: $\boldsymbol\theta \leftarrow \boldsymbol\theta - \alpha_k(\hat{\mathbf{m}}/(\sqrt{\hat{\mathbf{v}}} + \epsilon) + \lambda\boldsymbol\theta)$ | §7.3 |

Everything here is first-order: the parameter count ($10^8$–$10^{12}$) rules out anything that touches a Hessian. The "second-order" information comes only through Adam's diagonal scaling and through architectural conditioning (§11.3).

### 12.2 SVM — Duality & the Kernel Trick

**Primal (hard margin):** find the widest-margin separating hyperplane.

$$\min_{\mathbf{w}, b} \tfrac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^\top\mathbf{x}_i + b) \ge 1,\; i = 1..N$$

A convex QP (§10.2) with $n + 1$ variables and $N$ inequality constraints.

**Lagrangian** with multipliers $\mu_i \ge 0$ (constraint written as $1 - y_i(\mathbf{w}^\top\mathbf{x}_i + b) \le 0$):

$$\mathcal{L} = \tfrac{1}{2}\|\mathbf{w}\|^2 - \sum_i \mu_i\bigl[y_i(\mathbf{w}^\top\mathbf{x}_i + b) - 1\bigr]$$

**Stationarity** (KKT #1) gives closed forms for the primal variables:

$$\nabla_{\mathbf{w}}\mathcal{L} = \mathbf{0} \;\Rightarrow\; \mathbf{w} = \sum_i \mu_i y_i \mathbf{x}_i, \qquad \partial_b\mathcal{L} = 0 \;\Rightarrow\; \sum_i \mu_i y_i = 0$$

**Substitute back** to get the dual (§3.4):

$$\max_{\boldsymbol\mu \ge 0} \sum_i \mu_i - \tfrac{1}{2}\sum_{i,j}\mu_i\mu_j y_i y_j\, \mathbf{x}_i^\top\mathbf{x}_j \quad \text{s.t.} \quad \sum_i \mu_i y_i = 0$$

Three things fall out of the KKT conditions:

| Observation | KKT source | Consequence |
|-------------|------------|-------------|
| $\mathbf{w}$ is a weighted sum of training points | Stationarity | The model is $\sum_i \mu_i y_i\, \mathbf{x}_i^\top\mathbf{x} + b$ |
| $\mu_i > 0$ only for points **on the margin** | Complementary slackness | Only **support vectors** matter; the rest can be deleted |
| Data enters only via $\mathbf{x}_i^\top \mathbf{x}_j$ | Form of the dual | Replace with $k(\mathbf{x}_i, \mathbf{x}_j)$ — the **kernel trick** — to get a nonlinear classifier without ever computing features |

Strong duality holds (convex QP, Slater satisfied when data is separable), so solving the dual solves the primal. **SMO** (§9.4) optimizes the dual two multipliers at a time in closed form. The soft-margin version adds the box constraint $0 \le \mu_i \le C$.

### 12.3 PCA as Constrained Optimization

**Problem:** find the unit direction along which centered data $\tilde X$ has maximum variance. With $\Sigma = \frac{1}{N}\tilde X^\top \tilde X$ (symmetric PSD):

$$\max_{\mathbf{w}} \mathbf{w}^\top \Sigma\, \mathbf{w} \quad \text{s.t.} \quad \|\mathbf{w}\|^2 = 1$$

**Proof that the answer is the top eigenvector.**

**Setup.** Constraint $g(\mathbf{w}) = \mathbf{w}^\top\mathbf{w} - 1 = 0$; Lagrangian $\mathcal{L} = \mathbf{w}^\top\Sigma\mathbf{w} - \lambda(\mathbf{w}^\top\mathbf{w} - 1)$.

**Step 1 — stationarity.**
Using $\nabla_{\mathbf{w}}(\mathbf{w}^\top \Sigma \mathbf{w}) = 2\Sigma\mathbf{w}$ for symmetric $\Sigma$ ([linear_algebra.md](linear_algebra.md) §12.2):

$$\nabla_{\mathbf{w}}\mathcal{L} = 2\Sigma\mathbf{w} - 2\lambda\mathbf{w} = \mathbf{0} \quad\Longrightarrow\quad \Sigma\mathbf{w} = \lambda\mathbf{w}$$

So every candidate is an **eigenvector** of $\Sigma$, and the multiplier is its eigenvalue.

**Step 2 — evaluate the objective at a candidate.**

$$\mathbf{w}^\top\Sigma\mathbf{w} = \mathbf{w}^\top(\lambda\mathbf{w}) = \lambda\,\|\mathbf{w}\|^2 = \lambda$$

**Conclusion.** The objective at each eigenvector equals its eigenvalue; the maximum is attained at the eigenvector with the **largest eigenvalue** $\lambda_{\max}$, and the maximum variance is $\lambda_{\max}$ itself. $\blacksquare$

The $k$-th component is the same problem with the added constraints $\mathbf{w} \perp \mathbf{w}_1, \ldots, \mathbf{w}_{k-1}$ — which by Proof 2 of [linear_algebra.md](linear_algebra.md) §2.5 is automatically satisfied by the $k$-th eigenvector. This is why PCA is "just" an eigendecomposition / SVD ([linear_algebra.md](linear_algebra.md) §15.3): the optimization problem has an exact algebraic solution.

### 12.4 Trajectory Optimization & MPC

Plan a robot's motion by minimizing cost over a horizon of $T$ steps subject to its dynamics:

$$\min_{\mathbf{x}_{0:T},\, \mathbf{u}_{0:T-1}} \sum_{t=0}^{T-1} \ell(\mathbf{x}_t, \mathbf{u}_t) + \ell_T(\mathbf{x}_T) \quad \text{s.t.} \quad \mathbf{x}_{t+1} = \mathbf{f}(\mathbf{x}_t, \mathbf{u}_t),\; \mathbf{x}_0 = \mathbf{x}_{\text{now}},\; \mathbf{u}_t \in \mathcal{U},\; \mathbf{x}_t \in \mathcal{X}$$

| Ingredient | Optimization concept |
|------------|---------------------|
| Dynamics $\mathbf{x}_{t+1} = \mathbf{f}(\mathbf{x}_t, \mathbf{u}_t)$ | Equality constraints — one Lagrange multiplier vector $\boldsymbol\lambda_t$ per time step, the **costate** (§3.2) |
| Actuator / joint limits | Box inequality constraints (§3.3, §9.1) |
| Obstacle avoidance | Non-convex inequality constraints, usually softened as penalties (§9.3) |
| Quadratic cost + linear dynamics (**LQR**) | Equality-constrained QP — solvable exactly by a Riccati recursion, the time-structured version of the KKT block system |
| Nonlinear dynamics (**iLQR / DDP**) | Gauss–Newton / Newton on the trajectory (§8.1, §8.3): linearize $\mathbf{f}$, solve the LQR sub-problem, repeat |
| Constraints + nonlinear (**SQP**) | Sequential QP: each iteration solves a QP built from the current linearization |

**Model Predictive Control (MPC):** solve this problem at every control tick, apply only $\mathbf{u}_0$, shift the horizon, repeat. Because consecutive problems are nearly identical, **warm-starting** an active-set or ADMM QP solver (§9.3, §10.2) from the previous solution makes kHz rates feasible on legged robots and drones.

### 12.5 Policy Gradient & Trust Regions

Reinforcement learning maximizes expected return $J(\boldsymbol\theta) = \mathbb{E}_{\tau \sim \pi_{\boldsymbol\theta}}[R(\tau)]$ over a **stochastic policy**. The gradient passes through the sampling distribution via the log-derivative trick:

$$\nabla_{\boldsymbol\theta} J = \mathbb{E}_{\tau \sim \pi_{\boldsymbol\theta}}\left[\sum_t \nabla_{\boldsymbol\theta}\log\pi_{\boldsymbol\theta}(a_t \mid s_t)\, \hat{A}_t\right]$$

This is stochastic gradient **ascent** (§6.1) with extremely high-variance estimates — hence baselines, advantage estimation, and large batches.

**Why plain gradient steps are dangerous here:** a step in $\boldsymbol\theta$ that looks small can change the *policy distribution* a lot, and a bad policy produces bad data for the next step — a feedback loop GD on a fixed dataset never faces. The fix is to constrain the step in **distribution space**, not parameter space:

| Method | Constraint | Optimization tool |
|--------|-----------|-------------------|
| **Natural policy gradient** | Step $\Delta$ with $\frac{1}{2}\Delta^\top F\Delta \le \delta$ | Natural gradient $F^{-1}\nabla J$ (§8.4) |
| **TRPO** | $\mathbb{E}[\text{KL}(\pi_{\text{old}} \,\|\, \pi_{\boldsymbol\theta})] \le \delta$ | Trust region; solve $F\Delta = \nabla J$ with conjugate gradient + line search |
| **PPO** | Clip the probability ratio $\frac{\pi_{\boldsymbol\theta}}{\pi_{\text{old}}}$ to $[1 - \epsilon, 1 + \epsilon]$ | First-order surrogate of the trust region — Adam on the clipped objective |

PPO is the workhorse for RLHF fine-tuning of language models: the same AdamW loop as §12.1, with the clipped surrogate standing in for the trust-region constraint and a KL penalty to the reference model as an augmented-Lagrangian-style soft constraint (§9.3).

---

## Summary — What to Master First

| Priority | Topic | Because |
|----------|-------|---------|
| 1 | Convexity, local = global, first-order condition (§2, §3.1) | Tells you whether an optimizer's answer means anything |
| 2 | Gradient descent, step size, condition number (§4) | Every training loop; explains why learning rate and normalization matter |
| 3 | SGD, momentum, Adam/AdamW, schedules (§5–7) | The actual optimizers used in practice |
| 4 | Lagrange multipliers, KKT, duality (§3.2–3.4) | SVMs, PCA, constrained control, RL with constraints, any "s.t." |
| 5 | Newton, Gauss–Newton, LM, L-BFGS (§8) | Robotics, calibration, SLAM, small deterministic problems |
| 6 | Proximal / projected methods, QP (§9–10) | Sparsity, gradient clipping, MPC |

**Companion documents:** [multivariable_calculus.md](multivariable_calculus.md) (gradients, Hessians, Taylor expansion, KKT introduction) · [linear_algebra.md](linear_algebra.md) (PD matrices, eigendecomposition, least squares, KKT block system) · [statistics_probability.md](statistics_probability.md) (MLE, KL divergence, Monte Carlo estimation)
