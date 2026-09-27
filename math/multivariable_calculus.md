# Multivariable Calculus for AI & Robotics

Multivariable calculus is how we reason about change in many dimensions at once. Training a neural network is gradient descent on a scalar loss over millions of parameters; controlling a robot arm is inverting a Jacobian that maps joint velocities to end-effector velocities. Both are multivariable calculus.

## Table of Contents

1. [Functions of Several Variables](#1-functions-of-several-variables)
   - 1.1 [Scalar Fields & Vector Fields](#11-scalar-fields--vector-fields)
   - 1.2 [Level Sets & Contours](#12-level-sets--contours)
2. [Partial Derivatives](#2-partial-derivatives)
   - 2.1 [Definition](#21-definition)
   - 2.2 [Higher-Order Partials & Clairaut's Theorem](#22-higher-order-partials--clairauts-theorem)
   - 2.3 [The Gradient](#23-the-gradient)
3. [Directional Derivatives](#3-directional-derivatives)
   - 3.1 [Definition](#31-definition)
   - 3.2 [Gradient as Steepest Ascent](#32-gradient-as-steepest-ascent)
4. [The Chain Rule](#4-the-chain-rule)
   - 4.1 [Single Path](#41-single-path)
   - 4.2 [Multiple Paths — Computational Graphs](#42-multiple-paths--computational-graphs)
   - 4.3 [Backpropagation as the Chain Rule](#43-backpropagation-as-the-chain-rule)
5. [Linear Approximation & Taylor Expansion](#5-linear-approximation--taylor-expansion)
   - 5.1 [Tangent Plane](#51-tangent-plane)
   - 5.2 [Total Differential](#52-total-differential)
   - 5.3 [Multivariable Taylor Expansion](#53-multivariable-taylor-expansion)
6. [The Jacobian](#6-the-jacobian)
   - 6.1 [Definition](#61-definition)
   - 6.2 [Geometric Meaning — Local Linear Map](#62-geometric-meaning--local-linear-map)
   - 6.3 [Change of Variables & the Jacobian Determinant](#63-change-of-variables--the-jacobian-determinant)
   - 6.4 [Robotics: The Manipulator Jacobian](#64-robotics-the-manipulator-jacobian)
7. [The Hessian & Second-Order Behavior](#7-the-hessian--second-order-behavior)
   - 7.1 [Definition](#71-definition)
   - 7.2 [Curvature & the Loss Landscape](#72-curvature--the-loss-landscape)
   - 7.3 [Quadratic Approximation](#73-quadratic-approximation)
8. [Unconstrained Optimization](#8-unconstrained-optimization)
   - 8.1 [Critical Points](#81-critical-points)
   - 8.2 [Second-Derivative Test](#82-second-derivative-test)
   - 8.3 [Gradient Descent & Newton's Method](#83-gradient-descent--newtons-method)
9. [Constrained Optimization](#9-constrained-optimization)
   - 9.1 [Equality Constraints — Lagrange Multipliers](#91-equality-constraints--lagrange-multipliers)
   - 9.2 [Inequality Constraints — KKT Conditions](#92-inequality-constraints--kkt-conditions)
   - 9.3 [Lagrangian Duality](#93-lagrangian-duality)
10. [Vector Calculus Operators](#10-vector-calculus-operators)
    - 10.1 [Divergence](#101-divergence)
    - 10.2 [Curl](#102-curl)
    - 10.3 [Laplacian](#103-laplacian)
11. [Multiple Integrals](#11-multiple-integrals)
    - 11.1 [Double & Triple Integrals](#111-double--triple-integrals)
    - 11.2 [Change of Variables in Integrals](#112-change-of-variables-in-integrals)
    - 11.3 [Expectation as an Integral](#113-expectation-as-an-integral)
12. [AI & Robotics Applications](#12-ai--robotics-applications)
    - 12.1 [Backpropagation](#121-backpropagation)
    - 12.2 [Normalizing Flows](#122-normalizing-flows)
    - 12.3 [Inverse Kinematics](#123-inverse-kinematics)
    - 12.4 [The Reparameterization Trick](#124-the-reparameterization-trick)

---

## 1. Functions of Several Variables

### 1.1 Scalar Fields & Vector Fields

A **scalar field** assigns one number to each point in space; a **vector field** assigns a vector.

$$f: \mathbb{R}^n \to \mathbb{R} \qquad \text{(scalar field)} \qquad\qquad \mathbf{F}: \mathbb{R}^n \to \mathbb{R}^m \qquad \text{(vector field)}$$

| Object | Example in AI / Robotics |
|--------|--------------------------|
| Scalar field $f: \mathbb{R}^n \to \mathbb{R}$ | Loss $\mathcal{L}(\boldsymbol\theta)$ as a function of all weights; potential field for path planning |
| Vector field $\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^n$ | Gradient $\nabla\mathcal{L}$; velocity field of a fluid; attractive/repulsive forces in potential-field navigation |
| Map $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ | Forward kinematics: joint angles → end-effector pose; a neural network layer |

### 1.2 Level Sets & Contours

A **level set** is where the function is constant:

$$S_c = \{ \mathbf{x} \in \mathbb{R}^n \mid f(\mathbf{x}) = c \}$$

In 2D these are **contour lines** (like a topographic map); in 3D, **level surfaces**.

> **Key fact:** $\nabla f$ is always **perpendicular** to the level set through that point. Walking along a contour keeps $f$ constant, so the rate of change in that direction is zero — meaning $\nabla f$ has no component along the contour.

This is exactly why constrained optimization works (§9): at an optimum on a constraint curve, the objective's gradient must be parallel to the constraint's gradient.

---

## 2. Partial Derivatives

### 2.1 Definition

The **partial derivative** of $f$ with respect to $x_i$ is the ordinary derivative taken while holding every other variable fixed:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(\dots, x_i + h, \dots) - f(\dots, x_i, \dots)}{h}$$

**Example:**

$$f(x, y) = x^2 y + \sin(y) \qquad \Rightarrow \qquad \frac{\partial f}{\partial x} = 2xy, \qquad \frac{\partial f}{\partial y} = x^2 + \cos(y)$$

**Concrete numbers** at $(x, y) = (3, 0)$:

$$\frac{\partial f}{\partial x} = 2(3)(0) = 0, \qquad \frac{\partial f}{\partial y} = 3^2 + \cos(0) = 10$$

Moving in $x$ does not change $f$ here; moving in $y$ changes it at rate 10.

### 2.2 Higher-Order Partials & Clairaut's Theorem

Second partials come in two kinds — pure ($\partial^2 f / \partial x_i^2$) and mixed ($\partial^2 f / \partial x_i \partial x_j$).

$$\text{From } f = x^2 y + \sin y: \quad \frac{\partial^2 f}{\partial x^2} = 2y, \quad \frac{\partial^2 f}{\partial y^2} = -\sin y, \quad \frac{\partial^2 f}{\partial x\, \partial y} = 2x = \frac{\partial^2 f}{\partial y\, \partial x}$$

> **Clairaut's / Schwarz's theorem:** if the second partials are continuous, the order of differentiation does not matter: $\dfrac{\partial^2 f}{\partial x_i \partial x_j} = \dfrac{\partial^2 f}{\partial x_j \partial x_i}$.

This is why the **Hessian is symmetric** (§7) — which in turn guarantees it has real eigenvalues and orthogonal eigenvectors.

### 2.3 The Gradient

The **gradient** collects all first partials into a vector:

$$\nabla f(\mathbf{x}) = \begin{bmatrix} \partial f / \partial x_1 \\ \partial f / \partial x_2 \\ \vdots \\ \partial f / \partial x_n \end{bmatrix} \in \mathbb{R}^n$$

| Property | Meaning |
|----------|---------|
| Direction of $\nabla f$ | Direction of **steepest ascent** of $f$ |
| $\|\nabla f\|$ | The slope in that steepest direction |
| $\nabla f = \mathbf{0}$ | A **critical point** — candidate minimum, maximum, or saddle |
| $\nabla f \perp$ level set | Gradient is normal to the contour/surface |

**In AI:** training is $\boldsymbol\theta \leftarrow \boldsymbol\theta - \alpha \nabla_{\boldsymbol\theta}\mathcal{L}$. The whole enterprise is estimating one gradient vector per step.

---

## 3. Directional Derivatives

### 3.1 Definition

The rate of change of $f$ as you move from $\mathbf{x}$ in the direction of a **unit vector** $\mathbf{u}$:

$$D_{\mathbf{u}} f(\mathbf{x}) = \lim_{h \to 0} \frac{f(\mathbf{x} + h\mathbf{u}) - f(\mathbf{x})}{h} = \nabla f(\mathbf{x})^\top \mathbf{u}$$

The directional derivative is just the **dot product of the gradient with the direction**.

**Example:** $f(x,y) = x^2 + 3y^2$ at $(1, 2)$, direction $\mathbf{u} = \frac{1}{\sqrt2}[1, 1]^\top$.

$$\nabla f = [2x,\ 6y]^\top = [2,\ 12]^\top, \qquad D_{\mathbf{u}} f = [2,\ 12] \cdot \tfrac{1}{\sqrt2}[1,\ 1]^\top = \frac{14}{\sqrt2} \approx 9.9$$

### 3.2 Gradient as Steepest Ascent

Since $D_{\mathbf{u}} f = \nabla f^\top \mathbf{u} = \|\nabla f\| \cos\theta$, it is:

| Direction $\mathbf{u}$ | $D_{\mathbf{u}} f$ |
|------------------------|--------------------|
| Aligned with $\nabla f$ ($\theta = 0$) | Maximal: $+\|\nabla f\|$ — **steepest ascent** |
| Opposite to $\nabla f$ ($\theta = \pi$) | Minimal: $-\|\nabla f\|$ — **steepest descent** |
| Perpendicular to $\nabla f$ ($\theta = \pi/2$) | Zero — moving along the level set |

> **This is the justification for gradient descent:** $-\nabla f$ is provably the direction that decreases $f$ fastest, locally.

---

## 4. The Chain Rule

### 4.1 Single Path

If $z = f(x, y)$ and both $x = x(t)$, $y = y(t)$ depend on a parameter $t$:

$$\frac{dz}{dt} = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt}$$

Sum the contribution of each variable: (how $z$ depends on $x$) × (how $x$ depends on $t$), plus the same through $y$.

**Example:** $z = x^2 y$, with $x = \cos t$, $y = \sin t$.

$$\frac{dz}{dt} = (2xy)(-\sin t) + (x^2)(\cos t) = -2\cos t \sin^2 t + \cos^3 t$$

### 4.2 Multiple Paths — Computational Graphs

For $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ composed with $\mathbf{g}: \mathbb{R}^m \to \mathbb{R}^p$, the chain rule is **matrix multiplication of Jacobians**:

$$\mathbf{h} = \mathbf{g}(\mathbf{f}(\mathbf{x})) \qquad \Rightarrow \qquad J_{\mathbf{h}} = J_{\mathbf{g}} \, J_{\mathbf{f}}$$

$$\underbrace{\frac{\partial \mathbf{h}}{\partial \mathbf{x}}}_{p \times n} = \underbrace{\frac{\partial \mathbf{h}}{\partial \mathbf{f}}}_{p \times m} \cdot \underbrace{\frac{\partial \mathbf{f}}{\partial \mathbf{x}}}_{m \times n}$$

When multiple paths connect a variable to the output, their Jacobian products **add** — this is the multivariable chain rule, and it is the mathematical content of backpropagation.

### 4.3 Backpropagation as the Chain Rule

A network is a composition $\mathcal{L} = \ell \circ f_L \circ \cdots \circ f_1$. The loss gradient w.r.t. an early activation is a product of layer Jacobians:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{a}_1} = J_{f_2}^\top J_{f_3}^\top \cdots J_{f_L}^\top \, \nabla_{\mathbf{a}_L}\ell$$

| Direction | What it computes | Cost |
|-----------|------------------|------|
| **Forward-mode** (right-to-left in $J$ products) | One input's effect on all outputs | Cheap when inputs ≪ outputs |
| **Reverse-mode** (left-to-right) = backprop | All inputs' effect on one scalar output | Cheap when outputs ≪ inputs — the ML case (one scalar loss, millions of params) |

> **Key insight:** backprop is reverse-mode automatic differentiation. It is efficient precisely because the loss is a **single scalar**, so we propagate one vector backward instead of a full Jacobian.

For the matrix mechanics — per-layer Jacobians, shape tables, and fully worked numerical examples (linear layer, ReLU, softmax, MSE) — see [linear_algebra.md](linear_algebra.md) §12.

---

## 5. Linear Approximation & Taylor Expansion

### 5.1 Tangent Plane

Near a point $\mathbf{x}_0$, a differentiable function is well-approximated by its **linearization**:

$$f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top (\mathbf{x} - \mathbf{x}_0)$$

This is the equation of the tangent plane (or hyperplane). Every gradient step implicitly trusts this approximation over a small region — the "trust region."

### 5.2 Total Differential

The infinitesimal change in $f$ from small changes $d\mathbf{x}$:

$$df = \nabla f^\top d\mathbf{x} = \frac{\partial f}{\partial x_1}dx_1 + \cdots + \frac{\partial f}{\partial x_n}dx_n$$

**In robotics:** if $\mathbf{p} = \text{FK}(\boldsymbol\theta)$ is forward kinematics, then $d\mathbf{p} = J(\boldsymbol\theta)\, d\boldsymbol\theta$ — small joint motions map to small end-effector motions through the Jacobian.

### 5.3 Multivariable Taylor Expansion

To second order:

$$f(\mathbf{x}_0 + \Delta\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top \Delta\mathbf{x} + \frac{1}{2}\Delta\mathbf{x}^\top H(\mathbf{x}_0)\, \Delta\mathbf{x}$$

| Term | Order | Role |
|------|-------|------|
| $f(\mathbf{x}_0)$ | 0 | Value at the base point |
| $\nabla f^\top \Delta\mathbf{x}$ | 1 | Linear slope — used by gradient descent |
| $\frac12 \Delta\mathbf{x}^\top H \Delta\mathbf{x}$ | 2 | Curvature — used by Newton's method, trust-region methods, and the second-derivative test |

---

## 6. The Jacobian

### 6.1 Definition

For a vector-valued map $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the **Jacobian** is the matrix of all first partials:

$$J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix}
\dfrac{\partial f_1}{\partial x_1} & \cdots & \dfrac{\partial f_1}{\partial x_n} \\[2ex]
\vdots & \ddots & \vdots \\[1ex]
\dfrac{\partial f_m}{\partial x_1} & \cdots & \dfrac{\partial f_m}{\partial x_n}
\end{bmatrix} \in \mathbb{R}^{m \times n}$$

Row $i$ is $\nabla f_i^\top$ — the gradient of the $i$-th output component.

> **Layout note:** this document uses **numerator layout** ($m \times n$: outputs down the rows, inputs across the columns) — the standard in robotics and analysis, where $J$ is read directly as the local linear map $\mathbf{f}(\mathbf{x}_0 + \Delta\mathbf{x}) \approx \mathbf{f}(\mathbf{x}_0) + J\Delta\mathbf{x}$. The companion [linear_algebra.md](linear_algebra.md) §11.2, §12 uses **denominator layout** ($n \times m$) so ML gradients match parameter shapes. The two are transposes; each doc is internally consistent.

### 6.2 Geometric Meaning — Local Linear Map

The Jacobian is the **best linear approximation** of $\mathbf{f}$ at a point:

$$\mathbf{f}(\mathbf{x}_0 + \Delta\mathbf{x}) \approx \mathbf{f}(\mathbf{x}_0) + J(\mathbf{x}_0)\, \Delta\mathbf{x}$$

Whatever nonlinear thing $\mathbf{f}$ does, zoomed in far enough it acts like the matrix $J$ — stretching, rotating, and shearing a tiny neighborhood.

### 6.3 Change of Variables & the Jacobian Determinant

When $\mathbf{f}$ maps $\mathbb{R}^n \to \mathbb{R}^n$, $|\det J|$ is the **local volume scaling factor** — the nonlinear-map version of the determinant-as-volume fact for constant matrices ([linear_algebra.md](linear_algebra.md) §5).

$$dV_{\mathbf{y}} = |\det J|\; dV_{\mathbf{x}}$$

| $\det J$ | Meaning |
|----------|---------|
| $|\det J| > 1$ | Map expands volume locally |
| $|\det J| < 1$ | Map contracts volume locally |
| $\det J = 0$ | Map collapses a dimension — **singular**, not locally invertible |
| $\det J \neq 0$ | **Inverse Function Theorem:** $\mathbf{f}$ is locally invertible near that point |

**Example — polar coordinates** $(x, y) = (r\cos\theta,\ r\sin\theta)$:

$$J = \begin{bmatrix} \cos\theta & -r\sin\theta \\ \sin\theta & r\cos\theta \end{bmatrix}, \qquad \det J = r\cos^2\theta + r\sin^2\theta = r$$

Hence the familiar $dx\, dy = r\, dr\, d\theta$.

### 6.4 Robotics: The Manipulator Jacobian

For a robot arm, forward kinematics maps joint angles $\boldsymbol\theta \in \mathbb{R}^n$ to end-effector pose $\mathbf{x} \in \mathbb{R}^6$ (position + orientation). Its Jacobian relates **velocities**:

$$\dot{\mathbf{x}} = J(\boldsymbol\theta)\, \dot{\boldsymbol\theta}, \qquad
J = \begin{bmatrix} J_v \\ J_\omega \end{bmatrix} \in \mathbb{R}^{6 \times n}$$

| Quantity | Relation | Use |
|----------|----------|-----|
| Velocity | $\dot{\mathbf{x}} = J\dot{\boldsymbol\theta}$ | Resolved-rate motion control |
| Force / torque | $\boldsymbol\tau = J^\top \mathbf{F}$ | Static force mapping, impedance control (virtual work) |
| Inverse kinematics | $\dot{\boldsymbol\theta} = J^{+}\dot{\mathbf{x}}$ | $J^{+}$ = Moore–Penrose pseudo-inverse (see [linear_algebra.md](linear_algebra.md) §7.4) |

**Singularities:** when $\det(JJ^\top) = 0$ the arm loses a degree of freedom — some Cartesian direction becomes unreachable instantaneously, and $J^{+}$ blows up. The **manipulability measure** $w = \sqrt{\det(JJ^\top)}$ quantifies how far the configuration is from a singularity.

> **The transpose duality** $\boldsymbol\tau = J^\top \mathbf{F}$ comes from the principle of virtual work: $\mathbf{F}^\top \delta\mathbf{x} = \mathbf{F}^\top J \,\delta\boldsymbol\theta = \boldsymbol\tau^\top \delta\boldsymbol\theta$ for all $\delta\boldsymbol\theta$.

---

## 7. The Hessian & Second-Order Behavior

### 7.1 Definition

For a scalar function $f: \mathbb{R}^n \to \mathbb{R}$, the **Hessian** is the matrix of second partials:

$$H = \nabla^2 f = \begin{bmatrix}
\dfrac{\partial^2 f}{\partial x_1^2} & \cdots & \dfrac{\partial^2 f}{\partial x_1 \partial x_n} \\[2ex]
\vdots & \ddots & \vdots \\[1ex]
\dfrac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \dfrac{\partial^2 f}{\partial x_n^2}
\end{bmatrix} \in \mathbb{R}^{n \times n}$$

By Clairaut's theorem (§2.2) it is **symmetric** — so all eigenvalues are real and eigenvectors are orthogonal.

The Hessian is the Jacobian of the gradient: $H = J_{\nabla f}$.

### 7.2 Curvature & the Loss Landscape

The eigenvalues of $H$ are the curvatures along its eigenvector directions — real, with orthogonal eigenvectors, since $H$ is symmetric ([linear_algebra.md](linear_algebra.md) §2.5).

| Hessian at a critical point | Shape | Meaning |
|-----------------------------|-------|---------|
| All $\lambda_i > 0$ (PD) | Bowl | **Local minimum** |
| All $\lambda_i < 0$ (ND) | Dome | **Local maximum** |
| Mixed signs | Saddle | **Saddle point** — up in some directions, down in others |
| Some $\lambda_i = 0$ | Flat valley | Test is inconclusive (degenerate) |

| Condition number $\kappa = \lambda_{\max}/\lambda_{\min}$ | Effect on training |
|---------------------------------------------------------|--------------------|
| $\kappa \approx 1$ | Round bowl — gradient descent converges fast |
| $\kappa \gg 1$ | Long narrow ravine — gradient descent zig-zags; needs momentum or preconditioning |

> **In deep learning:** high-dimensional loss surfaces are dominated by **saddle points**, not local minima. This is why plain gradient descent stalls and why momentum / Adam help — they push through the near-flat directions of a saddle.

### 7.3 Quadratic Approximation

Near a critical point ($\nabla f = 0$), the function looks purely quadratic:

$$f(\mathbf{x}_0 + \Delta\mathbf{x}) \approx f(\mathbf{x}_0) + \frac{1}{2}\Delta\mathbf{x}^\top H\, \Delta\mathbf{x}$$

Newton's method (§8.3) minimizes this local quadratic exactly in one step.

---

## 8. Unconstrained Optimization

### 8.1 Critical Points

$\mathbf{x}^\star$ is a **critical point** of $f$ if

$$\nabla f(\mathbf{x}^\star) = \mathbf{0}$$

This is the **first-order necessary condition** for a local extremum. Every minimum is a critical point; not every critical point is a minimum.

### 8.2 Second-Derivative Test

Classify a critical point by the definiteness of the Hessian there ([linear_algebra.md](linear_algebra.md) §10 for PD / PSD / indefinite):

| $H(\mathbf{x}^\star)$ | Conclusion |
|-----------------------|------------|
| Positive definite | Strict local **minimum** |
| Negative definite | Strict local **maximum** |
| Indefinite | **Saddle point** |
| Positive/negative semidefinite (singular) | Inconclusive |

**2D shortcut:** with $H = \begin{bmatrix} f_{xx} & f_{xy} \\ f_{xy} & f_{yy}\end{bmatrix}$, let $D = \det H = f_{xx}f_{yy} - f_{xy}^2$.

- $D > 0,\ f_{xx} > 0$ → minimum
- $D > 0,\ f_{xx} < 0$ → maximum
- $D < 0$ → saddle
- $D = 0$ → inconclusive

### 8.3 Gradient Descent & Newton's Method

| Method | Update | Uses | Trade-off |
|--------|--------|------|-----------|
| Gradient descent | $\mathbf{x} \leftarrow \mathbf{x} - \alpha \nabla f$ | 1st order | Cheap per step, many steps, sensitive to $\kappa$ |
| Newton's method | $\mathbf{x} \leftarrow \mathbf{x} - H^{-1}\nabla f$ | 2nd order | Quadratic convergence near optimum, but $H^{-1}$ is $O(n^3)$ |
| Quasi-Newton (BFGS, L-BFGS) | $\mathbf{x} \leftarrow \mathbf{x} - B^{-1}\nabla f$ | Approx. $H$ | Builds curvature estimate from gradient history |
| Adam / RMSProp | Per-coordinate scaled gradient | Diagonal 2nd-order proxy | Standard for deep nets — $n$ in the millions |

**Newton intuition:** set the gradient of the quadratic model (§7.3) to zero: $\nabla f + H\Delta\mathbf{x} = 0 \Rightarrow \Delta\mathbf{x} = -H^{-1}\nabla f$.

---

## 9. Constrained Optimization

### 9.1 Equality Constraints — Lagrange Multipliers

**Problem:** minimize $f(\mathbf{x})$ subject to $g(\mathbf{x}) = 0$.

At a constrained optimum, you cannot decrease $f$ while staying on the constraint surface. That happens exactly when the objective's gradient is **parallel** to the constraint's gradient:

$$\nabla f(\mathbf{x}^\star) = \lambda\, \nabla g(\mathbf{x}^\star)$$

$\lambda$ is the **Lagrange multiplier**. Equivalently, define the **Lagrangian**

$$\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) - \lambda\, g(\mathbf{x})$$

and solve $\nabla_{\mathbf{x}}\mathcal{L} = 0$, $\partial \mathcal{L} / \partial \lambda = 0$ (the latter just re-states $g(\mathbf{x}) = 0$).

**Geometric picture:** at the optimum the level set of $f$ is **tangent** to the constraint curve. If they crossed transversally, you could slide along the constraint to a lower level set.

**Worked example:** maximize $f(x,y) = xy$ subject to $x + y = 10$.

$$\nabla f = [y,\ x]^\top, \quad \nabla g = [1,\ 1]^\top \;\Rightarrow\; y = \lambda,\; x = \lambda \;\Rightarrow\; x = y$$

With $x + y = 10$: $x = y = 5$, giving $f = 25$. (Maximum area rectangle for a fixed perimeter is a square.)

**Multiple constraints:** with $g_1 = \cdots = g_k = 0$,

$$\nabla f = \sum_{i=1}^{k} \lambda_i \nabla g_i$$

**Meaning of $\lambda$ — the shadow price:** if the constraint is relaxed to $g(\mathbf{x}) = c$, then

$$\frac{\partial f^\star}{\partial c} = \lambda$$

$\lambda$ measures how much the optimal value improves per unit of loosening the constraint. Economists call this the shadow price; in SVMs the multipliers are exactly the support-vector weights.

### 9.2 Inequality Constraints — KKT Conditions

**Problem:** minimize $f(\mathbf{x})$ subject to $g_i(\mathbf{x}) \le 0$ and $h_j(\mathbf{x}) = 0$.

Form the generalized Lagrangian:

$$\mathcal{L}(\mathbf{x}, \boldsymbol\mu, \boldsymbol\lambda) = f(\mathbf{x}) + \sum_i \mu_i g_i(\mathbf{x}) + \sum_j \lambda_j h_j(\mathbf{x})$$

The **Karush–Kuhn–Tucker (KKT) conditions** are necessary for optimality (and sufficient when the problem is convex):

| Condition | Statement | Reading |
|-----------|-----------|---------|
| Stationarity | $\nabla_{\mathbf{x}}\mathcal{L} = \mathbf{0}$ | No feasible descent direction |
| Primal feasibility | $g_i(\mathbf{x}) \le 0,\; h_j(\mathbf{x}) = 0$ | The solution obeys the constraints |
| Dual feasibility | $\mu_i \ge 0$ | Inequality multipliers are non-negative |
| Complementary slackness | $\mu_i\, g_i(\mathbf{x}) = 0$ | Either a constraint is active ($g_i = 0$) or its multiplier is zero |

> **Complementary slackness** is the key idea: an inequality constraint either "matters" (is tight, $g_i = 0$, $\mu_i > 0$) or it is slack and can be ignored ($\mu_i = 0$). This is how SVMs end up depending only on the support vectors.

### 9.3 Lagrangian Duality

The **dual function** is the Lagrangian minimized over $\mathbf{x}$:

$$d(\boldsymbol\mu, \boldsymbol\lambda) = \min_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol\mu, \boldsymbol\lambda), \qquad \boldsymbol\mu \ge 0$$

| Property | Statement |
|----------|-----------|
| **Weak duality** | $d(\boldsymbol\mu, \boldsymbol\lambda) \le f(\mathbf{x}^\star)$ always — the dual lower-bounds the primal |
| **Strong duality** | For convex problems with a feasible interior point (Slater's condition), $\max d = \min f$ — no gap |
| **Why it helps** | The dual is often lower-dimensional or has simpler constraints; the SVM dual depends only on inner products $\mathbf{x}_i^\top \mathbf{x}_j$, which enables the **kernel trick** |

---

## 10. Vector Calculus Operators

These act on fields. $\nabla = [\partial_{x_1}, \dots, \partial_{x_n}]^\top$ is the "del" operator.

### 10.1 Divergence

Maps a vector field to a scalar field — the net **outflow** per unit volume (a source if positive, a sink if negative):

$$\nabla \cdot \mathbf{F} = \sum_{i=1}^n \frac{\partial F_i}{\partial x_i}$$

**In ML:** continuous normalizing flows track log-density change via $\frac{d}{dt}\log p(\mathbf{x}(t)) = -\nabla \cdot \mathbf{f}$, where $\mathbf{f}$ is the flow's velocity field. Divergence-free flows preserve density.

### 10.2 Curl

Maps a 3D vector field to a vector field — the local **rotation** (axis and rate of swirl):

$$\nabla \times \mathbf{F} = \begin{bmatrix}
\partial_y F_z - \partial_z F_y \\
\partial_z F_x - \partial_x F_z \\
\partial_x F_y - \partial_y F_x
\end{bmatrix}$$

A field with $\nabla \times \mathbf{F} = \mathbf{0}$ is **conservative** — it is the gradient of some scalar potential, $\mathbf{F} = \nabla \phi$. Potential-field robot navigation relies on this: the planner designs $\phi$, and the robot follows $-\nabla\phi$.

### 10.3 Laplacian

The divergence of the gradient — a scalar measuring how much a point's value differs from the average of its neighbors:

$$\nabla^2 f = \Delta f = \sum_{i=1}^n \frac{\partial^2 f}{\partial x_i^2} = \text{tr}(H)$$

| Appears in | Role |
|------------|------|
| Diffusion / heat equation $\partial_t u = \Delta u$ | Basis of diffusion models |
| Graph Laplacian $L = D - A$ | Discrete analog; spectral clustering, graph neural networks |
| Laplacian smoothing | Mesh processing, regularization |

---

## 11. Multiple Integrals

### 11.1 Double & Triple Integrals

Integration accumulates a function over a region of area or volume:

$$\iint_R f(x, y)\; dA, \qquad \iiint_V f(x, y, z)\; dV$$

Evaluated as **iterated integrals** (Fubini's theorem — integrate one variable at a time):

$$\iint_R f\, dA = \int_{a}^{b}\!\!\left( \int_{c}^{d} f(x,y)\; dy \right) dx$$

### 11.2 Change of Variables in Integrals

Switching from variables $\mathbf{x}$ to $\mathbf{u}$ via $\mathbf{x} = \mathbf{g}(\mathbf{u})$:

$$\int_{\Omega} f(\mathbf{x})\; d\mathbf{x} = \int_{\Omega'} f(\mathbf{g}(\mathbf{u}))\; \bigl|\det J_{\mathbf{g}}(\mathbf{u})\bigr|\; d\mathbf{u}$$

The $|\det J|$ factor corrects for how the transformation stretches or shrinks volume (§6.3). **This single formula is the mathematical core of normalizing flows** (§12.2).

### 11.3 Expectation as an Integral

Expectation of a function of a continuous random vector is an integral against the density:

$$\mathbb{E}_{\mathbf{x} \sim p}[\,\phi(\mathbf{x})\,] = \int \phi(\mathbf{x})\, p(\mathbf{x})\; d\mathbf{x}$$

Most ML objectives are expectations: the training loss is $\mathbb{E}_{(\mathbf{x},y)\sim \mathcal{D}}[\ell]$, the ELBO is an expectation over the variational posterior, policy-gradient RL maximizes $\mathbb{E}_{\tau \sim \pi}[R(\tau)]$. Since these integrals are intractable, we estimate them with **Monte Carlo** samples — and then need the gradient to pass through the sampling (§12.4). See [statistics_probability.md](statistics_probability.md) for distributions and estimators.

---

## 12. AI & Robotics Applications

### 12.1 Backpropagation

Reverse-mode autodiff (§4.3) applied to $\mathcal{L}(\boldsymbol\theta)$. Each layer stores its local Jacobian during the forward pass; the backward pass multiplies upstream gradients by $J^\top$ layer by layer. Cost is the same order as the forward pass — the reason deep learning is feasible at all. Worked matrix examples: [linear_algebra.md](linear_algebra.md) §12.

### 12.2 Normalizing Flows

Build a complex density by pushing a simple one (Gaussian $\mathbf{z}$) through an invertible map $\mathbf{x} = f_{\boldsymbol\theta}(\mathbf{z})$. The change-of-variables formula (§11.2) gives exact likelihood:

$$\log p_X(\mathbf{x}) = \log p_Z(\mathbf{z}) - \log \bigl| \det J_f(\mathbf{z}) \bigr|$$

Flow architectures (RealNVP, Glow) are designed so $J_f$ is triangular, making $\det J_f$ a cheap product of diagonal entries.

### 12.3 Inverse Kinematics

Given a desired end-effector path $\mathbf{x}_d(t)$, solve for joint motions. The Jacobian (§6.4) gives an iterative Newton-style update:

$$\Delta\boldsymbol\theta = J^{+}(\boldsymbol\theta)\, \bigl(\mathbf{x}_d - \text{FK}(\boldsymbol\theta)\bigr)$$

Near singularities, use the **damped least-squares (Levenberg–Marquardt)** inverse $J^\top(JJ^\top + \rho^2 I)^{-1}$ to keep joint velocities bounded. Redundant arms ($n > 6$) use the null-space projector $(I - J^{+}J)$ to pursue secondary goals — avoiding joint limits or obstacles — without disturbing the end effector.

### 12.4 The Reparameterization Trick

To backprop through $\mathbb{E}_{\mathbf{z}\sim\mathcal{N}(\boldsymbol\mu,\, \boldsymbol\sigma^2)}[\phi(\mathbf{z})]$ (as in a VAE), rewrite the sample as a deterministic function of the parameters plus parameter-free noise:

$$\mathbf{z} = \boldsymbol\mu + \boldsymbol\sigma \odot \boldsymbol\epsilon, \qquad \boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0}, I)$$

Now the randomness does not depend on $\boldsymbol\mu, \boldsymbol\sigma$, so the gradient moves inside the expectation:

$$\nabla_{\boldsymbol\mu,\boldsymbol\sigma}\, \mathbb{E}[\phi(\mathbf{z})] = \mathbb{E}_{\boldsymbol\epsilon}\bigl[\nabla_{\boldsymbol\mu,\boldsymbol\sigma}\, \phi(\boldsymbol\mu + \boldsymbol\sigma \odot \boldsymbol\epsilon)\bigr]$$

and the chain rule (§4) handles the rest. This is just differentiating through a change of variables.

---

## Summary — What to Master First

| Priority | Topic | Because |
|----------|-------|---------|
| 1 | Gradient, directional derivative, chain rule (§2–4) | Every training step; backprop |
| 2 | Jacobian — shapes, geometry, pseudo-inverse (§6) | Robot kinematics, flows, sensitivity analysis |
| 3 | Hessian, curvature, Taylor (§5, §7) | Optimizer behavior, saddle points, conditioning |
| 4 | Lagrange multipliers & KKT (§9) | Constrained control, SVMs, trajectory optimization, RL with constraints |
| 5 | Change of variables & expectation integrals (§11) | Normalizing flows, VAEs, Monte Carlo objectives |

**Companion documents:** [linear_algebra.md](linear_algebra.md) (matrix derivatives, SVD, pseudo-inverse) · [statistics_probability.md](statistics_probability.md) (distributions, expectation, information theory)
