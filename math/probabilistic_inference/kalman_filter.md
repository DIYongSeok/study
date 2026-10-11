# The Kalman Filter

A robot never knows its state exactly. Encoders slip, IMUs drift, GPS jumps, and the dynamics model is only approximate. **State estimation** fuses a model of how the state evolves with noisy measurements of it, producing both an estimate and an honest measure of its uncertainty. For linear systems with Gaussian noise the optimal solution is the **Kalman filter**; its nonlinear extensions (EKF, UKF) and relatives (information filter, particle filter) are the backbone of localization, sensor fusion, and SLAM.

This document derives everything from first principles — every equation comes with a proof.

**Prerequisites:**

- Multivariate Gaussians: affine maps, marginals, conditionals, and the linear-Gaussian model with proofs — [statistics_probability.md](../basics/statistics_probability.md) §4.3 (Key Properties 1–4)
- Schur complement and Woodbury identity — [linear_algebra.md](../basics/linear_algebra.md) §4.6
- State-space models and discretization — [ode_linear_systems.md](../engineering_math/ode_linear_systems.md) §5, §8
- Matrix derivatives of traces — [multivariable_calculus.md](../basics/multivariable_calculus.md) §6.5

**Notation:**

| Symbol | Meaning |
|--------|---------|
| $\mathbf{x}_k \in \mathbb{R}^n$ | True state at step $k$ |
| $\mathbf{z}_k \in \mathbb{R}^m$ | Measurement at step $k$; $\mathbf{z}_{1:k} = (\mathbf{z}_1, \ldots, \mathbf{z}_k)$ |
| $\hat{\mathbf{x}}_k^-$, $P_k^-$ | **Prior** (predicted) mean and covariance: given $\mathbf{z}_{1:k-1}$ |
| $\hat{\mathbf{x}}_k$, $P_k$ | **Posterior** (updated) mean and covariance: given $\mathbf{z}_{1:k}$ |
| $\mathbf{e}_k^- = \mathbf{x}_k - \hat{\mathbf{x}}_k^-$, $\mathbf{e}_k = \mathbf{x}_k - \hat{\mathbf{x}}_k$ | Prior and posterior estimation errors |
| $\mathbf{y}_k$, $S_k$ | Innovation and its covariance |
| $K_k$ | Kalman gain |

## Table of Contents

1. [The Estimation Problem](#1-the-estimation-problem)
   - 1.1 [The Linear–Gaussian State-Space Model](#11-the-lineargaussian-state-space-model)
   - 1.2 [What "Optimal" Means](#12-what-optimal-means)
2. [Bayesian Filtering](#2-bayesian-filtering)
   - 2.1 [The Predict–Update Recursion](#21-the-predictupdate-recursion)
   - 2.2 [Why Gaussians Make It Tractable](#22-why-gaussians-make-it-tractable)
3. [The Kalman Filter](#3-the-kalman-filter)
   - 3.1 [The Algorithm](#31-the-algorithm)
   - 3.2 [Derivation of the Prediction Step](#32-derivation-of-the-prediction-step)
   - 3.3 [Derivation of the Update Step](#33-derivation-of-the-update-step)
   - 3.4 [The Belief Stays Gaussian — Proof by Induction](#34-the-belief-stays-gaussian--proof-by-induction)
   - 3.5 [Reading the Equations](#35-reading-the-equations)
4. [Optimality Without Gaussians](#4-optimality-without-gaussians)
   - 4.1 [Error Covariance for an Arbitrary Gain](#41-error-covariance-for-an-arbitrary-gain)
   - 4.2 [The Optimal Gain](#42-the-optimal-gain)
   - 4.3 [Unbiasedness](#43-unbiasedness)
   - 4.4 [The Orthogonality Principle & White Innovations](#44-the-orthogonality-principle--white-innovations)
5. [Equivalent Forms & Numerics](#5-equivalent-forms--numerics)
   - 5.1 [Joseph Form](#51-joseph-form)
   - 5.2 [Information Filter](#52-information-filter)
   - 5.3 [Sequential Measurement Updates](#53-sequential-measurement-updates)
   - 5.4 [Numerical Hygiene](#54-numerical-hygiene)
6. [Steady State & Observability](#6-steady-state--observability)
   - 6.1 [The Covariance Does Not Depend on the Data](#61-the-covariance-does-not-depend-on-the-data)
   - 6.2 [The Discrete Algebraic Riccati Equation](#62-the-discrete-algebraic-riccati-equation)
   - 6.3 [Scalar Case — Closed Form & Convergence Proof](#63-scalar-case--closed-form--convergence-proof)
   - 6.4 [Observability](#64-observability)
7. [Worked Examples](#7-worked-examples)
   - 7.1 [Scalar Random Walk](#71-scalar-random-walk)
   - 7.2 [Constant-Velocity Tracking](#72-constant-velocity-tracking)
8. [Nonlinear Filtering](#8-nonlinear-filtering)
   - 8.1 [Extended Kalman Filter (EKF)](#81-extended-kalman-filter-ekf)
   - 8.2 [Unscented Kalman Filter (UKF)](#82-unscented-kalman-filter-ukf)
   - 8.3 [Particle Filter](#83-particle-filter)
   - 8.4 [Error-State & Invariant EKF](#84-error-state--invariant-ekf)
9. [Filtering in Practice](#9-filtering-in-practice)
   - 9.1 [Choosing Q and R](#91-choosing-q-and-r)
   - 9.2 [Consistency Tests: NIS & NEES](#92-consistency-tests-nis--nees)
   - 9.3 [Outlier Gating](#93-outlier-gating)
   - 9.4 [Implementation](#94-implementation)
10. [Robotics Applications](#10-robotics-applications)
    - 10.1 [IMU + Position Fusion](#101-imu--position-fusion)
    - 10.2 [The Complementary Filter Is a Steady-State Kalman Filter](#102-the-complementary-filter-is-a-steady-state-kalman-filter)
    - 10.3 [Toward SLAM](#103-toward-slam)

---

## 1. The Estimation Problem

### 1.1 The Linear–Gaussian State-Space Model

$$\mathbf{x}_k = F\,\mathbf{x}_{k-1} + B\,\mathbf{u}_k + \mathbf{w}_k, \qquad \mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, Q)$$

$$\mathbf{z}_k = H\,\mathbf{x}_k + \mathbf{v}_k, \qquad \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, R)$$

| Symbol | Shape | Meaning |
|--------|-------|---------|
| $F$ | $n \times n$ | State transition — the discretized dynamics $A_d = e^{AT}$ ([ode_linear_systems.md](../engineering_math/ode_linear_systems.md) §8.3) |
| $B$, $\mathbf{u}_k$ | $n \times p$, $p$ | Known input (commands, or a sensor used as input such as an accelerometer) |
| $Q \succeq 0$ | $n \times n$ | **Process noise** covariance — model error and unknown disturbances |
| $H$ | $m \times n$ | Measurement model — which combinations of state the sensor sees |
| $R \succ 0$ | $m \times m$ | **Measurement noise** covariance — sensor noise |

**Assumptions** (all are used in the proofs below):

1. Initial state $\mathbf{x}_0 \sim \mathcal{N}(\hat{\mathbf{x}}_0, P_0)$.
2. $\mathbf{w}_k$ and $\mathbf{v}_k$ are **white** (independent across time), zero-mean, and Gaussian.
3. $\mathbf{x}_0$, $\{\mathbf{w}_k\}$, $\{\mathbf{v}_k\}$ are mutually independent.
4. $F, B, H, Q, R$ are known (they may vary with $k$; subscripts are dropped for readability).

Consequences used repeatedly:

- **Markov property:** given $\mathbf{x}_{k-1}$, the state $\mathbf{x}_k$ is independent of everything earlier (it depends only on $\mathbf{x}_{k-1}$ and the fresh noise $\mathbf{w}_k$).
- **Conditionally independent measurements:** given $\mathbf{x}_k$, the measurement $\mathbf{z}_k$ is independent of all other states and measurements (it depends only on $\mathbf{x}_k$ and the fresh noise $\mathbf{v}_k$).

### 1.2 What "Optimal" Means

We want an estimate $\hat{\mathbf{x}} = \mathbf{g}(\mathbf{z}_{1:k})$ that minimizes the **mean squared error** $\mathbb{E}\|\mathbf{x}_k - \mathbf{g}(\mathbf{z}_{1:k})\|^2$.

**Theorem (the conditional mean is the MMSE estimator).** Among all functions $\mathbf{g}$ of the data $\mathbf{z}$, the mean squared error is minimized by $\mathbf{g}^*(\mathbf{z}) = \mathbb{E}[\mathbf{x} \mid \mathbf{z}]$.

*Proof.* Write $\mathbf{m}(\mathbf{z}) = \mathbb{E}[\mathbf{x} \mid \mathbf{z}]$. For any $\mathbf{g}$,

$$\mathbb{E}\|\mathbf{x} - \mathbf{g}\|^2 = \mathbb{E}\|\mathbf{x} - \mathbf{m}\|^2 + \mathbb{E}\|\mathbf{m} - \mathbf{g}\|^2 + 2\,\mathbb{E}\big[(\mathbf{x} - \mathbf{m})^\top(\mathbf{m} - \mathbf{g})\big]$$

The cross term vanishes by the tower property: conditioning on $\mathbf{z}$, the factor $\mathbf{m} - \mathbf{g}$ is fixed, and $\mathbb{E}[\mathbf{x} - \mathbf{m} \mid \mathbf{z}] = \mathbf{0}$. So $\mathbb{E}\|\mathbf{x} - \mathbf{g}\|^2 = \mathbb{E}\|\mathbf{x} - \mathbf{m}\|^2 + \mathbb{E}\|\mathbf{m} - \mathbf{g}\|^2 \geq \mathbb{E}\|\mathbf{x} - \mathbf{m}\|^2$, with equality iff $\mathbf{g} = \mathbf{m}$. $\blacksquare$

So the goal is to compute the **posterior distribution** $p(\mathbf{x}_k \mid \mathbf{z}_{1:k})$ — its mean is the optimal estimate and its covariance is the remaining uncertainty. The Kalman filter does this exactly, recursively, in closed form.

---

## 2. Bayesian Filtering

### 2.1 The Predict–Update Recursion

**Theorem.** Under the Markov and conditional-independence properties of §1.1, the posterior can be updated recursively:

$$\text{Predict:}\quad p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}) = \int p(\mathbf{x}_k \mid \mathbf{x}_{k-1})\;p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1})\;d\mathbf{x}_{k-1}$$

$$\text{Update:}\quad p(\mathbf{x}_k \mid \mathbf{z}_{1:k}) = \frac{p(\mathbf{z}_k \mid \mathbf{x}_k)\;p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1})}{p(\mathbf{z}_k \mid \mathbf{z}_{1:k-1})}$$

*Proof of predict* (Chapman–Kolmogorov). Marginalize over the previous state, then use the Markov property $p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{z}_{1:k-1}) = p(\mathbf{x}_k \mid \mathbf{x}_{k-1})$:

$$p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}) = \int p(\mathbf{x}_k, \mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1})\,d\mathbf{x}_{k-1} = \int p(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{z}_{1:k-1})\,p(\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1})\,d\mathbf{x}_{k-1}$$

*Proof of update.* Bayes' rule with everything conditioned on $\mathbf{z}_{1:k-1}$, then conditional independence $p(\mathbf{z}_k \mid \mathbf{x}_k, \mathbf{z}_{1:k-1}) = p(\mathbf{z}_k \mid \mathbf{x}_k)$:

$$p(\mathbf{x}_k \mid \mathbf{z}_k, \mathbf{z}_{1:k-1}) = \frac{p(\mathbf{z}_k \mid \mathbf{x}_k, \mathbf{z}_{1:k-1})\,p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1})}{p(\mathbf{z}_k \mid \mathbf{z}_{1:k-1})} \qquad \blacksquare$$

| Step | What happens to the belief | Uncertainty |
|------|---------------------------|-------------|
| Predict | Pushed through the dynamics and blurred by process noise | Grows |
| Update | Multiplied by the measurement likelihood and renormalized | Shrinks |

This recursion holds for **any** model satisfying the two properties — it is also the starting point of the EKF, UKF, and particle filter (§8).

### 2.2 Why Gaussians Make It Tractable

In general the integral and the normalization above have no closed form. For the linear-Gaussian model, both steps map Gaussians to Gaussians:

- Predict: an affine map of a Gaussian plus independent Gaussian noise — Key Property 1.
- Update: a Gaussian prior with a linear-Gaussian measurement — Key Property 4.

So the belief is always $\mathcal{N}(\hat{\mathbf{x}}, P)$, and the infinite-dimensional recursion reduces to updating **one vector and one matrix** per step.

---

## 3. The Kalman Filter

### 3.1 The Algorithm

Initialize with $\hat{\mathbf{x}}_0$, $P_0$. For $k = 1, 2, \ldots$:

**Predict**

$$\hat{\mathbf{x}}_k^- = F\,\hat{\mathbf{x}}_{k-1} + B\,\mathbf{u}_k$$

$$P_k^- = F\,P_{k-1}\,F^\top + Q$$

**Update**

$$\mathbf{y}_k = \mathbf{z}_k - H\,\hat{\mathbf{x}}_k^- \qquad \text{(innovation)}$$

$$S_k = H\,P_k^-\,H^\top + R \qquad \text{(innovation covariance)}$$

$$K_k = P_k^-\,H^\top S_k^{-1} \qquad \text{(Kalman gain)}$$

$$\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k^- + K_k\,\mathbf{y}_k$$

$$P_k = (I - K_kH)\,P_k^-$$

| Quantity | Shape |
|----------|-------|
| $\mathbf{y}_k$ | $m$ |
| $S_k$ | $m \times m$ |
| $K_k$ | $n \times m$ |

### 3.2 Derivation of the Prediction Step

**Claim.** If $\mathbf{x}_{k-1} \mid \mathbf{z}_{1:k-1} \sim \mathcal{N}(\hat{\mathbf{x}}_{k-1}, P_{k-1})$, then $\mathbf{x}_k \mid \mathbf{z}_{1:k-1} \sim \mathcal{N}(F\hat{\mathbf{x}}_{k-1} + B\mathbf{u}_k,\ FP_{k-1}F^\top + Q)$.

*Proof.* Conditioned on $\mathbf{z}_{1:k-1}$, the noise $\mathbf{w}_k$ is still $\mathcal{N}(\mathbf{0}, Q)$ and independent of $\mathbf{x}_{k-1}$ (it is a future noise; assumption 3). So $(\mathbf{x}_{k-1}, \mathbf{w}_k)$ is jointly Gaussian with covariance $\operatorname{diag}(P_{k-1}, Q)$, and

$$\mathbf{x}_k = \begin{bmatrix} F & I \end{bmatrix}\begin{bmatrix} \mathbf{x}_{k-1} \\ \mathbf{w}_k \end{bmatrix} + B\mathbf{u}_k$$

is an affine map of it. By Key Property 1 it is Gaussian with mean $F\hat{\mathbf{x}}_{k-1} + \mathbf{0} + B\mathbf{u}_k$ and covariance $\begin{bmatrix} F & I \end{bmatrix}\operatorname{diag}(P_{k-1}, Q)\begin{bmatrix} F & I \end{bmatrix}^\top = FP_{k-1}F^\top + Q$. $\blacksquare$

### 3.3 Derivation of the Update Step

**Claim.** If $\mathbf{x}_k \mid \mathbf{z}_{1:k-1} \sim \mathcal{N}(\hat{\mathbf{x}}_k^-, P_k^-)$, then $\mathbf{x}_k \mid \mathbf{z}_{1:k} \sim \mathcal{N}(\hat{\mathbf{x}}_k^- + K_k\mathbf{y}_k,\ (I - K_kH)P_k^-)$.

*Proof.* Conditioned on $\mathbf{z}_{1:k-1}$, we have a Gaussian prior on $\mathbf{x}_k$ and a measurement $\mathbf{z}_k = H\mathbf{x}_k + \mathbf{v}_k$ with $\mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, R)$ independent of $\mathbf{x}_k$ and of $\mathbf{z}_{1:k-1}$. This is exactly the linear-Gaussian model, Key Property 4, with $\boldsymbol\mu = \hat{\mathbf{x}}_k^-$, $P = P_k^-$, $\mathbf{c} = \mathbf{0}$. Its part (iii) gives

$$\mathbf{x}_k \mid \mathbf{z}_k, \mathbf{z}_{1:k-1} \sim \mathcal{N}\big(\hat{\mathbf{x}}_k^- + K_k(\mathbf{z}_k - H\hat{\mathbf{x}}_k^-),\ (I - K_kH)P_k^-\big), \qquad K_k = P_k^-H^\top(HP_k^-H^\top + R)^{-1} \qquad \blacksquare$$

For completeness, the core of that property is Gaussian conditioning: the joint prior of state and measurement is

$$\begin{bmatrix} \mathbf{x}_k \\ \mathbf{z}_k \end{bmatrix} \Big|\ \mathbf{z}_{1:k-1} \sim \mathcal{N}\!\left(\begin{bmatrix} \hat{\mathbf{x}}_k^- \\ H\hat{\mathbf{x}}_k^- \end{bmatrix}, \begin{bmatrix} P_k^- & P_k^-H^\top \\ HP_k^- & S_k \end{bmatrix}\right)$$

and conditioning on the second block (Key Property 3) gives mean $\hat{\mathbf{x}}_k^- + P_k^-H^\top S_k^{-1}(\mathbf{z}_k - H\hat{\mathbf{x}}_k^-)$ and covariance $P_k^- - P_k^-H^\top S_k^{-1}HP_k^-$. The Kalman gain is the **regression coefficient** of the state on the measurement.

**By-product — the predictive distribution of the measurement.** Part (ii) of Key Property 4: $\mathbf{z}_k \mid \mathbf{z}_{1:k-1} \sim \mathcal{N}(H\hat{\mathbf{x}}_k^-, S_k)$, i.e. the innovation $\mathbf{y}_k \sim \mathcal{N}(\mathbf{0}, S_k)$. This is the normalizer in Bayes' rule and the basis of consistency checks and gating (§9).

### 3.4 The Belief Stays Gaussian — Proof by Induction

**Theorem.** For every $k \geq 0$, $p(\mathbf{x}_k \mid \mathbf{z}_{1:k}) = \mathcal{N}(\hat{\mathbf{x}}_k, P_k)$ with $\hat{\mathbf{x}}_k, P_k$ produced by the algorithm of §3.1. Hence $\hat{\mathbf{x}}_k$ is the MMSE estimate (§1.2) and $P_k = \mathbb{E}[\mathbf{e}_k\mathbf{e}_k^\top \mid \mathbf{z}_{1:k}]$ is its exact error covariance.

*Proof.* Base case $k = 0$: assumption 1. Inductive step: if it holds at $k - 1$, §3.2 gives the Gaussian prior at $k$, and §3.3 turns it into the Gaussian posterior at $k$ with the stated parameters. $\blacksquare$

### 3.5 Reading the Equations

**The update is a weighted correction.** $\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k^- + K_k(\mathbf{z}_k - H\hat{\mathbf{x}}_k^-)$: start from the prediction and move toward what the measurement says, by an amount set by $K_k$.

**Scalar case** ($n = m = 1$, $H = 1$): $K = \dfrac{P^-}{P^- + R}$, so

$$\hat{x} = (1 - K)\,\hat{x}^- + K\,z, \qquad P = \frac{P^-R}{P^- + R} = \left(\frac{1}{P^-} + \frac{1}{R}\right)^{-1}$$

— a precision-weighted average, and precisions add.

| Limit | Gain | Behavior |
|-------|------|----------|
| $R \to 0$ (perfect sensor) | $K \to H^{-1}$ (when $H$ is invertible) | Trust the measurement completely |
| $R \to \infty$ (useless sensor) | $K \to 0$ | Ignore the measurement; pure prediction |
| $P^- \to 0$ (perfect prediction) | $K \to 0$ | Ignore the measurement |
| $Q$ large | $P^-$ large, $K$ large | Model distrusted; fast but noisy tracking |
| $Q$ small | $K$ small | Model trusted; smooth but slow to react |

**Uncertainty never increases in an update:** $P_k = P_k^- - P_k^-H^\top S_k^{-1}HP_k^- \preceq P_k^-$, since the subtracted term is PSD.

---

## 4. Optimality Without Gaussians

Section 3 shows the Kalman filter is the exact Bayesian filter for Gaussian noise. A second derivation shows it is also the **best linear estimator** whenever the noises merely have the stated means and covariances — Gaussianity is not needed.

### 4.1 Error Covariance for an Arbitrary Gain

Consider the update $\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k^- + K(\mathbf{z}_k - H\hat{\mathbf{x}}_k^-)$ with **any** gain $K$. Substituting $\mathbf{z}_k = H\mathbf{x}_k + \mathbf{v}_k$:

$$\mathbf{e}_k = \mathbf{x}_k - \hat{\mathbf{x}}_k = \mathbf{e}_k^- - K(H\mathbf{e}_k^- + \mathbf{v}_k) = (I - KH)\,\mathbf{e}_k^- - K\,\mathbf{v}_k$$

$\mathbf{e}_k^-$ depends only on $\mathbf{x}_0$, $\mathbf{w}_{1:k}$, $\mathbf{v}_{1:k-1}$, so it is uncorrelated with $\mathbf{v}_k$. Therefore

$$\boxed{\ P_k(K) = (I - KH)\,P_k^-\,(I - KH)^\top + K R K^\top\ }$$

This **Joseph form** is the error covariance for any gain, optimal or not.

### 4.2 The Optimal Gain

**Theorem.** $K^* = P_k^-H^\top S_k^{-1}$ minimizes $P_k(K)$ in the strongest possible sense: $P_k(K) - P_k(K^*) \succeq 0$ for every $K$. At the optimum, $P_k(K^*) = (I - K^*H)P_k^-$.

*Proof 1 — completing the square.* Expand the Joseph form, using $S = HP^-H^\top + R$:

$$P(K) = P^- - KHP^- - P^-H^\top K^\top + KSK^\top$$

Using $K^*S = P^-H^\top$ (so $KHP^- = KSK^{*\top}$ and $P^-H^\top K^\top = K^*SK^\top$), this equals

$$P(K) = \underbrace{P^- - K^*SK^{*\top}}_{P(K^*)} + (K - K^*)\,S\,(K - K^*)^\top$$

as can be checked by expanding the last term. Since $S \succ 0$, the last term is PSD and vanishes only at $K = K^*$. Finally $K^*SK^{*\top} = P^-H^\top S^{-1}HP^- = K^*HP^-$, so $P(K^*) = (I - K^*H)P^-$. $\blacksquare$

*Proof 2 — matrix calculus.* Minimize the total error variance $\operatorname{tr}P(K)$. Using $\operatorname{tr}(P^-H^\top K^\top) = \operatorname{tr}(KHP^-)$:

$$\operatorname{tr}P(K) = \operatorname{tr}P^- - 2\operatorname{tr}(KHP^-) + \operatorname{tr}(KSK^\top)$$

With the identities $\dfrac{\partial}{\partial X}\operatorname{tr}(XA) = A^\top$ and $\dfrac{\partial}{\partial X}\operatorname{tr}(XBX^\top) = X(B + B^\top)$ (verify entry-wise: $\operatorname{tr}(XA) = \sum_{ij}X_{ij}A_{ji}$ and $\operatorname{tr}(XBX^\top) = \sum_{ijk}X_{ij}B_{jk}X_{ik}$):

$$\frac{\partial\operatorname{tr}P}{\partial K} = -2\,P^-H^\top + 2\,KS = 0 \quad\Longrightarrow\quad K^* = P^-H^\top S^{-1}$$

The objective is a convex quadratic in $K$ (its Hessian is built from $S \succ 0$), so this stationary point is the minimum. $\blacksquare$

### 4.3 Unbiasedness

**Theorem.** If $\hat{\mathbf{x}}_0 = \mathbb{E}[\mathbf{x}_0]$, then $\mathbb{E}[\mathbf{e}_k^-] = \mathbb{E}[\mathbf{e}_k] = \mathbf{0}$ for all $k$, for any choice of gains.

*Proof.* Induction. $\mathbb{E}[\mathbf{e}_0] = \mathbf{0}$. If $\mathbb{E}[\mathbf{e}_{k-1}] = \mathbf{0}$, then $\mathbf{e}_k^- = \mathbf{x}_k - \hat{\mathbf{x}}_k^- = F\mathbf{e}_{k-1} + \mathbf{w}_k$ has mean $\mathbf{0}$, and $\mathbf{e}_k = (I - KH)\mathbf{e}_k^- - K\mathbf{v}_k$ has mean $\mathbf{0}$. $\blacksquare$

Combining §4.1–4.3: with only zero-mean, white noises of known covariance, the Kalman filter is the **best linear unbiased estimator (BLUE)** in this recursive form. With Gaussian noise it is the best estimator of **any** form (§3.4).

### 4.4 The Orthogonality Principle & White Innovations

**Theorem (orthogonality).** With the optimal gain, the posterior error is uncorrelated with every innovation so far: $\mathbb{E}[\mathbf{e}_k\mathbf{y}_j^\top] = 0$ for all $j \leq k$.

*Proof.* Induction on $k$.

- $j = k$: with $\mathbf{y}_k = H\mathbf{e}_k^- + \mathbf{v}_k$,

$$\mathbb{E}[\mathbf{e}_k\mathbf{y}_k^\top] = \mathbb{E}\big[((I - KH)\mathbf{e}_k^- - K\mathbf{v}_k)(H\mathbf{e}_k^- + \mathbf{v}_k)^\top\big] = (I - KH)P_k^-H^\top - KR = P_k^-H^\top - KS_k = 0$$

  exactly at $K = K^* = P_k^-H^\top S_k^{-1}$.
- $j < k$: $\mathbf{e}_k^- = F\mathbf{e}_{k-1} + \mathbf{w}_k$. The noises $\mathbf{w}_k, \mathbf{v}_k$ are future relative to $\mathbf{y}_j$ and uncorrelated with it, and $\mathbb{E}[\mathbf{e}_{k-1}\mathbf{y}_j^\top] = 0$ by the induction hypothesis. So $\mathbb{E}[\mathbf{e}_k^-\mathbf{y}_j^\top] = 0$ and hence $\mathbb{E}[\mathbf{e}_k\mathbf{y}_j^\top] = (I - KH)\cdot 0 - K\cdot 0 = 0$. $\blacksquare$

Geometrically, $\hat{\mathbf{x}}_k$ is the **orthogonal projection** of $\mathbf{x}_k$ onto the span of the data, and the error is perpendicular to that span — the same principle as least squares ([linear_algebra.md](../basics/linear_algebra.md) §9.2).

**Corollary (innovations are white).** $\mathbb{E}[\mathbf{y}_k] = \mathbf{0}$, $\mathbb{E}[\mathbf{y}_k\mathbf{y}_k^\top] = S_k$, and $\mathbb{E}[\mathbf{y}_k\mathbf{y}_j^\top] = 0$ for $j < k$.

*Proof.* $\mathbf{y}_k = H\mathbf{e}_k^- + \mathbf{v}_k$ has zero mean (§4.3) and covariance $HP_k^-H^\top + R = S_k$. For $j < k$: $\mathbb{E}[\mathbf{y}_k\mathbf{y}_j^\top] = H\,\mathbb{E}[\mathbf{e}_k^-\mathbf{y}_j^\top] + \mathbb{E}[\mathbf{v}_k\mathbf{y}_j^\top] = 0$, using the step above and the whiteness of $\mathbf{v}$. $\blacksquare$

**Why this matters in practice:** a correctly tuned filter produces innovations that look like white noise with covariance $S_k$. Correlated or oversized innovations are the main diagnostic of a wrong model or wrong $Q, R$ (§9.2).

---

## 5. Equivalent Forms & Numerics

### 5.1 Joseph Form

$$P_k = (I - K_kH)\,P_k^-\,(I - K_kH)^\top + K_kRK_k^\top$$

Algebraically equal to $(I - K_kH)P_k^-$ at the optimal gain (§4.2), but numerically superior: it is a sum of two PSD terms, so rounding errors cannot make $P_k$ lose symmetry or positive definiteness. The short form subtracts two nearly equal matrices and can go indefinite after many steps.

### 5.2 Information Filter

Work with the **information matrix** $Y = P^{-1}$ and **information vector** $\hat{\boldsymbol\eta} = P^{-1}\hat{\mathbf{x}}$.

**Update in information form:**

$$Y_k = Y_k^- + H^\top R^{-1}H, \qquad \hat{\boldsymbol\eta}_k = \hat{\boldsymbol\eta}_k^- + H^\top R^{-1}\mathbf{z}_k$$

*Proof.* Key Property 4(iv) with $\mathbf{c} = \mathbf{0}$: $P_k^{-1} = (P_k^-)^{-1} + H^\top R^{-1}H$ and $\hat{\mathbf{x}}_k = P_k\big((P_k^-)^{-1}\hat{\mathbf{x}}_k^- + H^\top R^{-1}\mathbf{z}_k\big)$; multiply the second by $P_k^{-1}$. (Part (iv) itself was proved with the Woodbury identity.) $\blacksquare$

The gain can likewise be written $K_k = P_kH^\top R^{-1}$ (push-through identity, [linear_algebra.md](../basics/linear_algebra.md) §4.6).

| | Covariance form | Information form |
|---|---|---|
| Update | Inverts $m \times m$ ($S_k$) | Simple **addition** |
| Predict | Simple | Needs $Y_k^- = (FY_{k-1}^{-1}F^\top + Q)^{-1}$ |
| Many sensors at once | Stacks into a big $S$ | Contributions just add |
| No prior information | Needs "$P_0 = \infty$" | $Y_0 = 0$ is natural |
| Basis of | Standard KF | Graph-based SLAM / factor graphs (sparse information matrices) |

### 5.3 Sequential Measurement Updates

If a measurement vector splits into independent parts — $R = \operatorname{diag}(R_1, R_2)$ — then updating with $\mathbf{z}^{(1)}$ and then with $\mathbf{z}^{(2)}$ gives exactly the same posterior as one joint update.

*Proof.* Independence of the noises makes $\mathbf{z}^{(1)}$ and $\mathbf{z}^{(2)}$ conditionally independent given $\mathbf{x}$, so

$$p(\mathbf{x} \mid \mathbf{z}^{(1)}, \mathbf{z}^{(2)}) \propto p(\mathbf{z}^{(2)} \mid \mathbf{x})\,p(\mathbf{z}^{(1)} \mid \mathbf{x})\,p(\mathbf{x}) \propto p(\mathbf{z}^{(2)} \mid \mathbf{x})\;p(\mathbf{x} \mid \mathbf{z}^{(1)})$$

— Bayes' rule applied twice. Both routes produce the same (Gaussian) posterior. $\blacksquare$

Practical benefits: scalar updates need no matrix inversion ($S$ is a number), and sensors arriving at different rates are handled naturally.

### 5.4 Numerical Hygiene

| Problem | Remedy |
|---------|--------|
| $P$ loses symmetry | Symmetrize after each step: $P \leftarrow \tfrac12(P + P^\top)$ |
| $P$ loses positive definiteness | Joseph form (§5.1); square-root filters propagating a Cholesky factor of $P$ |
| Explicit $S^{-1}$ | Solve $S K^\top = H P^-$ with a Cholesky factorization instead |
| Very different scales in the state (e.g. meters and radians) | Choose units so that variances are comparable |

---

## 6. Steady State & Observability

### 6.1 The Covariance Does Not Depend on the Data

The equations for $P_k^-$, $S_k$, $K_k$, $P_k$ never use $\mathbf{z}_k$ or $\mathbf{u}_k$. The whole covariance and gain sequence can be **computed offline** before any measurement arrives — a direct consequence of Gaussian conditioning (Key Property 3: the conditional covariance does not depend on the observed value).

### 6.2 The Discrete Algebraic Riccati Equation

Combining predict and update for the prior covariance gives the **Riccati recursion**:

$$P_{k+1}^- = F\Big(P_k^- - P_k^-H^\top\big(HP_k^-H^\top + R\big)^{-1}HP_k^-\Big)F^\top + Q$$

For a time-invariant model it typically converges to a constant $P_\infty^-$ solving the **discrete algebraic Riccati equation (DARE)**:

$$P_\infty^- = F\Big(P_\infty^- - P_\infty^-H^\top\big(HP_\infty^-H^\top + R\big)^{-1}HP_\infty^-\Big)F^\top + Q$$

with constant gain $K_\infty = P_\infty^-H^\top(HP_\infty^-H^\top + R)^{-1}$. The **steady-state Kalman filter** uses $K_\infty$ from the start — a fixed linear filter with no covariance bookkeeping, common on microcontrollers.

**Theorem (convergence, statement).** If $(F, H)$ is **detectable** (every unstable mode is observable) and $(F, Q^{1/2})$ is **stabilizable** (process noise excites every unstable mode), then the DARE has a unique PSD stabilizing solution, $P_k^- \to P_\infty^-$ from any $P_0 \succeq 0$, and the error dynamics $F(I - K_\infty H)$ are stable. The general proof uses monotonicity of the Riccati map and is lengthy (Anderson & Moore, *Optimal Filtering*, Ch. 4); the scalar case below shows the mechanism completely.

### 6.3 Scalar Case — Closed Form & Convergence Proof

Random walk: $F = H = 1$, noise variances $Q > 0$, $R > 0$. The prior variance obeys

$$P_{k+1}^- = g(P_k^-), \qquad g(p) = \frac{pR}{p + R} + Q$$

**Fixed point.** $p = \tfrac{pR}{p + R} + Q \iff p^2 - Qp - QR = 0$, whose positive root is

$$p_\infty = \frac{Q + \sqrt{Q^2 + 4QR}}{2}, \qquad K_\infty = \frac{p_\infty}{p_\infty + R}$$

**Theorem.** $P_k^- \to p_\infty$ for every $P_0 \geq 0$.

*Proof.* After one step $P_1^- = g(\cdot) \geq Q$, and $g$ maps $[Q, \infty)$ into itself. On this interval $g'(p) = \dfrac{R^2}{(p + R)^2} \leq \dfrac{R^2}{(Q + R)^2} < 1$, so by the mean value theorem $|g(a) - g(b)| \leq c\,|a - b|$ with $c = R^2/(Q + R)^2 < 1$. $g$ is a contraction on the complete set $[Q, \infty)$, and by the Banach fixed-point theorem the iterates converge geometrically to its unique fixed point $p_\infty$. $\blacksquare$

**Small-noise approximation:** if $Q \ll R$, then $p_\infty \approx \sqrt{QR}$ and $K_\infty \approx \sqrt{Q/R}$ — the gain depends only on the **ratio** of process to measurement noise.

**Actual example — what the $Q/R$ ratio does.** Estimating a slowly varying quantity (say a wheel's speed) from noisy samples, with $R = 1$:

| $Q/R$ | $K_\infty$ | Effective averaging window $\approx 1/K_\infty$ | Behavior |
|-------|-----------|------------------------------------------------|----------|
| $0.01$ | $0.095$ | $\approx 10$ samples | Very smooth, but lags behind real changes |
| $1$ | $0.618$ | $\approx 1.6$ samples | Moderate smoothing |
| $100$ | $0.990$ | $\approx 1$ sample | Just copies the latest measurement |

Choosing $Q$ is choosing how fast the filter is allowed to believe the world changes.

### 6.4 Observability

The filter can only reduce uncertainty in directions the measurements reveal (directly or through the dynamics). The pair $(F, H)$ is **observable** iff

$$\operatorname{rank}\begin{bmatrix} H \\ HF \\ \vdots \\ HF^{n-1} \end{bmatrix} = n$$

*Why:* with no noise, $\mathbf{z}_1, \ldots, \mathbf{z}_n$ equal this matrix times $\mathbf{x}_1$ (plus known input terms); $\mathbf{x}_1$ is recoverable iff the stacked matrix has full column rank ([linear_algebra.md](../basics/linear_algebra.md) §8.3). By Cayley–Hamilton, $F^n$ and higher powers add no new rows.

**Example** (constant velocity, state = position and velocity, $F = \begin{bmatrix} 1 & T \\ 0 & 1 \end{bmatrix}$):

| Sensor | $H$ | Stacked rows | Rank | Result |
|--------|-----|-------------|------|--------|
| Position | $[1\ \ 0]$ | $[1\ \ 0],\ [1\ \ T]$ | 2 | Observable — velocity inferred from changing position |
| Velocity only | $[0\ \ 1]$ | $[0\ \ 1],\ [0\ \ 1]$ | 1 | Position unobservable — its variance grows without bound |

---

## 7. Worked Examples

### 7.1 Scalar Random Walk

$x_k = x_{k-1} + w_k$, $z_k = x_k + v_k$, with $Q = 1$, $R = 2$, $\hat{x}_0 = 0$, $P_0 = 10$ (very uncertain start). Measurements $z = 1.2,\ 0.5,\ 1.8$:

| $k$ | $P_k^- = P_{k-1} + 1$ | $K_k = P_k^-/(P_k^- + 2)$ | $\hat{x}_k = \hat{x}_{k-1} + K_k(z_k - \hat{x}_{k-1})$ | $P_k = (1 - K_k)P_k^-$ |
|-----|----------|-------|------------|--------|
| 1 | 11.000 | 0.8462 | $0 + 0.8462(1.2) = 1.0154$ | 1.6923 |
| 2 | 2.6923 | 0.5738 | $1.0154 + 0.5738(0.5 - 1.0154) = 0.7197$ | 1.1475 |
| 3 | 2.1475 | 0.5178 | $0.7197 + 0.5178(1.8 - 0.7197) = 1.2791$ | 1.0356 |

The first measurement is trusted heavily (the prior was vague); the gain then settles quickly. Steady state from §6.3: $p_\infty = \tfrac{1 + \sqrt{1 + 8}}{2} = 2$, $K_\infty = \tfrac{2}{2 + 2} = 0.5$, $P_\infty = 1$ — check: $P_\infty + Q = 2 = p_\infty$ ✓.

### 7.2 Constant-Velocity Tracking

State $\mathbf{x} = (p, v)$, sample time $T$, position measured. Model the unknown acceleration as continuous white noise of intensity $q$ entering the velocity.

**Process noise derivation.** With $A = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}$ and noise input $G = (0, 1)^\top$, the discretized noise covariance is ([ode_linear_systems.md](../engineering_math/ode_linear_systems.md) §9.3)

$$Q = q\int_0^T e^{A\sigma}GG^\top e^{A^\top\sigma}\,d\sigma, \qquad e^{A\sigma}G = \begin{bmatrix} 1 & \sigma \\ 0 & 1 \end{bmatrix}\begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} \sigma \\ 1 \end{bmatrix}$$

$$Q = q\int_0^T \begin{bmatrix} \sigma^2 & \sigma \\ \sigma & 1 \end{bmatrix}d\sigma = q\begin{bmatrix} T^3/3 & T^2/2 \\ T^2/2 & T \end{bmatrix}, \qquad F = \begin{bmatrix} 1 & T \\ 0 & 1 \end{bmatrix}, \qquad H = \begin{bmatrix} 1 & 0 \end{bmatrix}$$

**One step with numbers:** $T = 0.1$ s, $q = 1$, $R = 0.25$ (σ = 0.5 m), $\hat{\mathbf{x}}_0 = (0, 1)$, $P_0 = I$, measurement $z_1 = 0.3$.

*Predict:*

$$\hat{\mathbf{x}}_1^- = \begin{bmatrix} 0.1 \\ 1 \end{bmatrix}, \qquad P_1^- = FP_0F^\top + Q = \begin{bmatrix} 1.01 & 0.1 \\ 0.1 & 1 \end{bmatrix} + \begin{bmatrix} 0.000333 & 0.005 \\ 0.005 & 0.1 \end{bmatrix} = \begin{bmatrix} 1.01033 & 0.105 \\ 0.105 & 1.1 \end{bmatrix}$$

*Update:*

$$y_1 = 0.3 - 0.1 = 0.2, \qquad S_1 = 1.01033 + 0.25 = 1.26033, \qquad K_1 = \frac{1}{1.26033}\begin{bmatrix} 1.01033 \\ 0.105 \end{bmatrix} = \begin{bmatrix} 0.80164 \\ 0.08331 \end{bmatrix}$$

$$\hat{\mathbf{x}}_1 = \begin{bmatrix} 0.1 + 0.80164 \cdot 0.2 \\ 1 + 0.08331 \cdot 0.2 \end{bmatrix} = \begin{bmatrix} 0.26033 \\ 1.01666 \end{bmatrix}, \qquad P_1 = (I - K_1H)P_1^- = \begin{bmatrix} 0.20041 & 0.02083 \\ 0.02083 & 1.09125 \end{bmatrix}$$

The position measurement also nudged the **velocity**, through the correlation $0.105$ that the prediction step created between position and velocity. That correlation is how an unmeasured state gets corrected.

---

## 8. Nonlinear Filtering

Model: $\mathbf{x}_k = \mathbf{f}(\mathbf{x}_{k-1}, \mathbf{u}_k) + \mathbf{w}_k$, $\mathbf{z}_k = \mathbf{h}(\mathbf{x}_k) + \mathbf{v}_k$. The Bayesian recursion of §2.1 still holds, but a nonlinear map of a Gaussian is not Gaussian, so the belief must be approximated.

### 8.1 Extended Kalman Filter (EKF)

**Idea:** linearize $\mathbf{f}$ and $\mathbf{h}$ around the current estimate ([ode_linear_systems.md](../engineering_math/ode_linear_systems.md) §6.2) and apply the linear Kalman filter to the error.

**Derivation.** First-order Taylor expansion of the dynamics around $\hat{\mathbf{x}}_{k-1}$:

$$\mathbf{x}_k \approx \mathbf{f}(\hat{\mathbf{x}}_{k-1}, \mathbf{u}_k) + F_k(\mathbf{x}_{k-1} - \hat{\mathbf{x}}_{k-1}) + \mathbf{w}_k, \qquad F_k = \frac{\partial\mathbf{f}}{\partial\mathbf{x}}\bigg|_{\hat{\mathbf{x}}_{k-1}, \mathbf{u}_k}$$

This is affine in $\mathbf{x}_{k-1}$, so §3.2 applies: mean $\mathbf{f}(\hat{\mathbf{x}}_{k-1}, \mathbf{u}_k)$, covariance $F_kP_{k-1}F_k^\top + Q$. Similarly $\mathbf{h}(\mathbf{x}_k) \approx \mathbf{h}(\hat{\mathbf{x}}_k^-) + H_k(\mathbf{x}_k - \hat{\mathbf{x}}_k^-)$ with $H_k = \partial\mathbf{h}/\partial\mathbf{x}|_{\hat{\mathbf{x}}_k^-}$, and §3.3 applies with the predicted measurement $\mathbf{h}(\hat{\mathbf{x}}_k^-)$.

**Algorithm:**

$$\hat{\mathbf{x}}_k^- = \mathbf{f}(\hat{\mathbf{x}}_{k-1}, \mathbf{u}_k), \qquad P_k^- = F_kP_{k-1}F_k^\top + Q$$

$$\mathbf{y}_k = \mathbf{z}_k - \mathbf{h}(\hat{\mathbf{x}}_k^-), \quad S_k = H_kP_k^-H_k^\top + R, \quad K_k = P_k^-H_k^\top S_k^{-1}, \quad \hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k^- + K_k\mathbf{y}_k, \quad P_k = (I - K_kH_k)P_k^-$$

Note: the **mean** is propagated through the true nonlinear functions; only the **covariance** uses the Jacobians.

**Example — range measurement.** A beacon at the origin measures distance $h(\mathbf{x}) = \sqrt{p_x^2 + p_y^2} = r$:

$$H = \frac{\partial h}{\partial(p_x, p_y)} = \begin{bmatrix} \dfrac{p_x}{r} & \dfrac{p_y}{r} \end{bmatrix}$$

— a unit vector pointing from the beacon to the robot: a range sensor informs only the **radial** direction.

**Actual example — one EKF update with a range beacon.** Prior: robot at $\hat{\mathbf{x}}^- = (3, 4)$ m with $P^- = I$ (1 m std in every direction). The beacon at the origin measures $z = 5.5$ m with $R = 0.01$ (10 cm std).

| Quantity | Value |
|----------|-------|
| Predicted range $h(\hat{\mathbf{x}}^-)$ | $\sqrt{3^2 + 4^2} = 5$ |
| Jacobian $H$ | $(3/5,\ 4/5) = (0.6,\ 0.8)$ |
| Innovation $y$ | $5.5 - 5 = 0.5$ |
| $S = HP^-H^\top + R$ | $0.36 + 0.64 + 0.01 = 1.01$ |
| $K = P^-H^\top/S$ | $(0.594,\ 0.792)$ |
| Updated position $\hat{\mathbf{x}}^- + Ky$ | $(3.297,\ 4.396)$, now at range $5.495$ |
| Updated covariance $(I - KH)P^-$ | $\begin{bmatrix} 0.644 & -0.475 \\ -0.475 & 0.366 \end{bmatrix}$ |

Project the new covariance onto two directions: along the beacon direction $(0.6, 0.8)$ the variance is $0.0099$ (std ≈ 10 cm — as good as the sensor), while perpendicular to it, $(-0.8, 0.6)$, it is still exactly $1.0$. The robot moved straight outward along the ray and became certain about its **distance**, but learned nothing about its **bearing**. A second beacon in a different direction is needed to pin down the position — the observability idea of §6.4 in action.

**Limitations:**

- **Bias:** $\mathbb{E}[\mathbf{f}(\mathbf{x})] \neq \mathbf{f}(\mathbb{E}[\mathbf{x}])$. To second order, $\mathbb{E}[f_i(\mathbf{x})] \approx f_i(\hat{\mathbf{x}}) + \tfrac12\operatorname{tr}\big(\nabla^2 f_i\,P\big)$ — the EKF drops the curvature term.
- **Overconfidence and divergence:** if $P$ is large compared to the region where the linearization is valid, the computed covariance is too small, and the filter can lock onto a wrong estimate.
- Jacobians must be derived (or auto-differentiated) correctly.
- **Iterated EKF:** re-linearize $\mathbf{h}$ at the updated estimate and repeat the update a few times — a Gauss–Newton step on the MAP problem ([optimization.md](../basics/optimization.md) §8.3).

### 8.2 Unscented Kalman Filter (UKF)

**Idea:** instead of linearizing the function, approximate the **distribution** with a few deterministically chosen **sigma points**, push them through the exact nonlinearity, and refit a Gaussian.

**Sigma points** for $\mathcal{N}(\boldsymbol\mu, P)$ in $\mathbb{R}^n$, with a matrix square root $LL^\top = (n + \lambda)P$ (e.g. Cholesky) and columns $\mathbf{l}_i$:

$$\boldsymbol\chi_0 = \boldsymbol\mu, \qquad \boldsymbol\chi_i = \boldsymbol\mu + \mathbf{l}_i, \qquad \boldsymbol\chi_{i+n} = \boldsymbol\mu - \mathbf{l}_i \qquad (i = 1, \ldots, n)$$

$$W_0 = \frac{\lambda}{n + \lambda}, \qquad W_i = W_{i+n} = \frac{1}{2(n + \lambda)}$$

**Theorem.** The weighted sample mean and covariance of the sigma points equal $\boldsymbol\mu$ and $P$ exactly.

*Proof.* The weights sum to one: $\tfrac{\lambda}{n + \lambda} + 2n\cdot\tfrac{1}{2(n + \lambda)} = 1$. The $\pm\mathbf{l}_i$ cancel in pairs, so $\sum_j W_j\boldsymbol\chi_j = \boldsymbol\mu$. For the covariance, $\boldsymbol\chi_0 - \boldsymbol\mu = \mathbf{0}$ and

$$\sum_{j} W_j(\boldsymbol\chi_j - \boldsymbol\mu)(\boldsymbol\chi_j - \boldsymbol\mu)^\top = \sum_{i=1}^{n}\frac{2}{2(n + \lambda)}\,\mathbf{l}_i\mathbf{l}_i^\top = \frac{1}{n + \lambda}LL^\top = P \qquad \blacksquare$$

**Corollary.** For an affine map $\mathbf{g}(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$, the transformed points $A\boldsymbol\chi_j + \mathbf{b}$ have weighted mean $A\boldsymbol\mu + \mathbf{b}$ and weighted covariance $APA^\top$ — exactly Key Property 1. So the UKF reduces to the Kalman filter for linear models, and for nonlinear ones it captures the mean correctly to second order without any Jacobians.

**Algorithm** (additive noise; in practice a separate covariance weight $W_0^{(c)} = W_0 + (1 - \alpha^2 + \beta)$ and $\lambda = \alpha^2(n + \kappa) - n$ are used to tune the spread; with $W_0^{(c)} = W_0$ everything below is exact for linear models):

1. **Predict:** sigma points of $(\hat{\mathbf{x}}_{k-1}, P_{k-1})$ → $\boldsymbol\chi_j^- = \mathbf{f}(\boldsymbol\chi_j, \mathbf{u}_k)$; $\hat{\mathbf{x}}_k^- = \sum W_j\boldsymbol\chi_j^-$; $P_k^- = \sum W_j(\boldsymbol\chi_j^- - \hat{\mathbf{x}}_k^-)(\cdot)^\top + Q$.
2. **Predicted measurements:** redraw sigma points from $(\hat{\mathbf{x}}_k^-, P_k^-)$, map $\mathbf{Z}_j = \mathbf{h}(\boldsymbol\chi_j)$; $\hat{\mathbf{z}} = \sum W_j\mathbf{Z}_j$.
3. **Covariances:** $S_k = \sum W_j(\mathbf{Z}_j - \hat{\mathbf{z}})(\cdot)^\top + R$, cross-covariance $C_k = \sum W_j(\boldsymbol\chi_j - \hat{\mathbf{x}}_k^-)(\mathbf{Z}_j - \hat{\mathbf{z}})^\top$.
4. **Update:** $K_k = C_kS_k^{-1}$, $\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_k^- + K_k(\mathbf{z}_k - \hat{\mathbf{z}})$, $P_k = P_k^- - K_kS_kK_k^\top$.

**Consistency with the Kalman filter.** For linear $\mathbf{h}(\mathbf{x}) = H\mathbf{x}$, the corollary gives $C_k = P_k^-H^\top$ and $S_k = HP_k^-H^\top + R$, so $K_k = P_k^-H^\top S_k^{-1}$ and $P_k^- - K_kS_kK_k^\top = P_k^- - P_k^-H^\top S_k^{-1}HP_k^- = (I - K_kH)P_k^-$ — identical to §3.1. The form $K = C S^{-1}$ is simply Gaussian conditioning (Key Property 3) with the cross-covariance estimated from sigma points.

### 8.3 Particle Filter

For strongly nonlinear or multimodal beliefs (global localization, kidnapped robot), represent the belief by $N$ weighted samples $\{(\mathbf{x}^{(i)}, w^{(i)})\}$.

**Derivation (sequential importance sampling).** Target: $p(\mathbf{x}_k \mid \mathbf{z}_{1:k}) \propto p(\mathbf{z}_k \mid \mathbf{x}_k)\,p(\mathbf{x}_k \mid \mathbf{z}_{1:k-1})$ (§2.1). If the particles are propagated through the motion model — $\mathbf{x}_k^{(i)} \sim p(\mathbf{x}_k \mid \mathbf{x}_{k-1}^{(i)})$ — they are samples from the predicted belief. Importance sampling ([statistics_probability.md](../basics/statistics_probability.md) §9.2) with proposal = prediction and target = posterior gives weights proportional to target/proposal $= p(\mathbf{z}_k \mid \mathbf{x}_k^{(i)})$:

$$w_k^{(i)} \propto w_{k-1}^{(i)}\,p(\mathbf{z}_k \mid \mathbf{x}_k^{(i)}), \qquad \sum_i w_k^{(i)} = 1$$

**Resampling.** Weights degenerate (one particle takes all the weight). Monitor the **effective sample size** $N_{\text{eff}} = 1/\sum_i (w^{(i)})^2$ (equal to $N$ for uniform weights, $1$ for a single dominant particle) and resample when it drops below, say, $N/2$.

Cost grows exponentially with state dimension, so particle filters are used for low-dimensional states or combined with Kalman filters (Rao–Blackwellized particle filters, FastSLAM).

### 8.4 Error-State & Invariant EKF

Robot orientation lives on a curved space (rotations, $SO(3)$), not in $\mathbb{R}^n$; adding a correction $K\mathbf{y}$ to a rotation matrix breaks it. The **error-state EKF** estimates a small error in a local vector-space parameterization and folds it back into the nominal state after each update. The **invariant EKF** goes further by choosing errors that respect the symmetry of the Lie group, which makes the linearized error dynamics independent of the estimate — giving much better consistency for IMU-based legged-robot estimation (e.g., the CoCo-InEKF article in this repository). Both require Lie group tools — the next stage after this document.

---

## 9. Filtering in Practice

### 9.1 Choosing Q and R

| Matrix | How to get it | If too small | If too large |
|--------|--------------|--------------|--------------|
| $R$ | Sensor datasheet; variance of a log recorded while stationary | Filter chases measurement noise | Ignores good measurements; slow |
| $Q$ | Physical reasoning (e.g. unmodeled acceleration, §7.2); tune on data | Overconfident; slow to react; can **diverge** when the model is wrong | Noisy estimate; little smoothing |

Since the gain depends mainly on the **ratio** $Q/R$ (§6.3), tune that ratio first, then scale both until the consistency tests below pass.

### 9.2 Consistency Tests: NIS & NEES

**Normalized innovation squared:** $\epsilon_k = \mathbf{y}_k^\top S_k^{-1}\mathbf{y}_k$.

**Theorem.** If the model is correct, $\epsilon_k \sim \chi^2_m$ (chi-squared with $m$ degrees of freedom), so $\mathbb{E}[\epsilon_k] = m$.

*Proof.* $\mathbf{y}_k \sim \mathcal{N}(\mathbf{0}, S_k)$ (§3.3 by-product). Let $\mathbf{u} = S_k^{-1/2}\mathbf{y}_k$; by Key Property 1, $\mathbf{u} \sim \mathcal{N}(\mathbf{0}, S_k^{-1/2}S_kS_k^{-1/2}) = \mathcal{N}(\mathbf{0}, I)$, so its components are independent standard normals (Lemma B). Then $\epsilon_k = \mathbf{u}^\top\mathbf{u} = \sum_{i=1}^m u_i^2$ is by definition $\chi^2_m$, with mean $m \cdot \mathbb{E}[u_i^2] = m$. $\blacksquare$

| Average NIS over a run | Diagnosis |
|------------------------|-----------|
| $\approx m$ | Consistent |
| $\gg m$ | Filter overconfident — $Q$ or $R$ too small, or model error |
| $\ll m$ | Filter underconfident — $Q$ or $R$ too large |

NIS needs only the filter's own outputs, so it can be monitored on the real robot. **NEES** $\mathbf{e}_k^\top P_k^{-1}\mathbf{e}_k \sim \chi^2_n$ is the same test on the state error, available in simulation where ground truth is known. Also check that innovations are uncorrelated over time (§4.4).

### 9.3 Outlier Gating

Before an update, compute $\epsilon_k$. A measurement whose NIS exceeds a chi-squared threshold is implausible under the model and is rejected:

| $m$ | 95% threshold | 99% threshold |
|-----|--------------|--------------|
| 1 | 3.84 | 6.63 |
| 2 | 5.99 | 9.21 |
| 3 | 7.81 | 11.34 |

This is a Mahalanobis-distance test ([statistics_probability.md](../basics/statistics_probability.md) §4.3) and the standard way to reject GPS multipath jumps, wrong visual feature matches, or contact misdetections. It is also used for **data association**: match each measurement to the landmark with the smallest NIS.

### 9.4 Implementation

```python
import numpy as np

class KalmanFilter:
    def __init__(self, F, H, Q, R, x0, P0, B=None):
        self.F, self.H, self.Q, self.R, self.B = F, H, Q, R, B
        self.x, self.P = x0.astype(float), P0.astype(float)

    def predict(self, u=None):
        self.x = self.F @ self.x + (self.B @ u if u is not None else 0)
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, gate=None):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        nis = y @ np.linalg.solve(S, y)
        if gate is not None and nis > gate:           # outlier rejection (§9.3)
            return nis
        K = np.linalg.solve(S, self.H @ self.P).T     # K = P H^T S^-1 without forming S^-1
        I_KH = np.eye(len(self.x)) - K @ self.H
        self.x = self.x + K @ y
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T   # Joseph form (§5.1)
        self.P = 0.5 * (self.P + self.P.T)
        return nis
```

---

## 10. Robotics Applications

### 10.1 IMU + Position Fusion

A common pattern uses the **accelerometer as an input** and a slower absolute sensor (GPS, motion capture, visual odometry) as the **measurement**. In 1D, with state $(p, v)$ and measured acceleration $a_k$:

$$\mathbf{x}_k = \begin{bmatrix} 1 & T \\ 0 & 1 \end{bmatrix}\mathbf{x}_{k-1} + \begin{bmatrix} T^2/2 \\ T \end{bmatrix}a_k + \mathbf{w}_k, \qquad z_k = \begin{bmatrix} 1 & 0 \end{bmatrix}\mathbf{x}_k + v_k$$

The input matrix is the exact ZOH discretization of the double integrator ([ode_linear_systems.md](../engineering_math/ode_linear_systems.md) §8.3). Accelerometer noise of variance $\sigma_a^2$ enters through the same matrix, so $Q = \sigma_a^2\,\mathbf{b}\mathbf{b}^\top$ with $\mathbf{b} = (T^2/2, T)$.

Run the prediction at the IMU rate (e.g. 500 Hz) and the update whenever a position measurement arrives (e.g. 10 Hz). Between updates the uncertainty grows; each update pulls it back. Adding the accelerometer **bias** as a third state (modeled as a slow random walk) lets the filter estimate and remove it — it is observable from the position measurements.

### 10.2 The Complementary Filter Is a Steady-State Kalman Filter

The IMU tilt complementary filter of [signals_systems.md](../engineering_math/signals_systems.md) §9.1 is

$$\hat\theta_k = \alpha\big(\hat\theta_{k-1} + \omega_kT\big) + (1 - \alpha)\,\theta_{\text{acc},k}$$

Now take the Kalman model: gyro as input, accelerometer angle as measurement,

$$\theta_k = \theta_{k-1} + T\omega_k + w_k, \qquad z_k = \theta_{\text{acc},k} = \theta_k + v_k$$

This is a scalar random walk with input. The Kalman update with steady-state gain $K_\infty$ is

$$\hat\theta_k = \underbrace{(\hat\theta_{k-1} + T\omega_k)}_{\hat\theta_k^-} + K_\infty\big(z_k - \hat\theta_k^-\big) = (1 - K_\infty)\big(\hat\theta_{k-1} + T\omega_k\big) + K_\infty\,\theta_{\text{acc},k}$$

— the complementary filter with $\alpha = 1 - K_\infty$. **Example:** gyro integration noise $Q = 10^{-6}\ \text{rad}^2$ per step and accelerometer noise $R = (0.05\ \text{rad})^2 = 2.5 \times 10^{-3}$ give, by §6.3, $K_\infty = 0.0198 \approx 0.02$ — the classic $\alpha = 0.98$. The Kalman view explains where the magic number comes from: it is $1 - \sqrt{Q/R}$ approximately, set by the two sensors' noise levels.

### 10.3 Toward SLAM

| Method | Idea | Builds on |
|--------|------|-----------|
| EKF-SLAM | Stack robot pose and all landmark positions into one state; EKF over all of it | §8.1; covariance grows as (number of landmarks)² |
| Information-form / graph SLAM | Information matrix is sparse; solve as nonlinear least squares over all poses (factor graphs) | §5.2; Gauss–Newton ([optimization.md](../basics/optimization.md) §8.3) |
| FastSLAM | Particles for the robot path, small EKFs for each landmark | §8.3 + §8.1 |
| Visual-inertial odometry (MSCKF, invariant filters) | Error-state EKF on Lie groups with IMU prediction and camera updates | §8.4 |

---

## Summary — What to Master First

| Priority | Topic | Because |
|----------|-------|---------|
| 1 | Gaussian conditioning and the linear-Gaussian model ([statistics_probability.md](../basics/statistics_probability.md) §4.3) | Every Kalman equation is one of these properties |
| 2 | The algorithm and its derivation (§3) | Predict = affine map; update = conditioning |
| 3 | Joseph form, minimum-variance gain, orthogonality (§4, §5.1) | Why the filter is optimal and how to keep it numerically healthy |
| 4 | Choosing $Q, R$; NIS consistency and gating (§9) | What actually makes a filter work on a robot |
| 5 | EKF and UKF (§8.1–8.2) | Real robots are nonlinear |
| 6 | Steady state, Riccati equation, observability (§6) | Fixed-gain filters; knowing what can be estimated at all |

**Companion documents:** [gaussian_process_regression.md](gaussian_process_regression.md) (the same Gaussian conditioning, applied to functions) · [statistics_probability.md](../basics/statistics_probability.md) (Gaussian properties with proofs) · [linear_algebra.md](../basics/linear_algebra.md) (Schur complement, Woodbury) · [ode_linear_systems.md](../engineering_math/ode_linear_systems.md) (state-space models, discretization) · [signals_systems.md](../engineering_math/signals_systems.md) (complementary filter, frequency view)
