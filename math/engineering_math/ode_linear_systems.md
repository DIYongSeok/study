# Ordinary Differential Equations & Linear Systems

Dynamics and control are about how a **state changes over time**. A robot joint accelerates under torque, a drone tilts under rotor thrust, an estimator's uncertainty grows between measurements — each is an ordinary differential equation (ODE). This document builds the toolkit in the order you need it for robotics: solve scalar ODEs by hand, solve forced problems with the **Laplace transform**, rewrite any linear ODE as a **state-space model** $\dot{\mathbf{x}} = A\mathbf{x} + B\mathbf{u}$, approximate nonlinear robots by **linearization**, decide **stability** from eigenvalues, and turn continuous models into the **discrete-time** updates that run on a computer.

**Prerequisites:** eigenvalues and the matrix exponential ([linear_algebra.md](../basics/linear_algebra.md) §6, §14.4), Jacobians ([multivariable_calculus.md](../basics/multivariable_calculus.md) §6, §6.5).

**Scope:** modeling and analysis. Frequency-domain tools (Fourier transform, frequency response, z-transform, filters) are in [signals_systems.md](signals_systems.md). Controller design (controllability, pole placement, LQR, PID tuning) builds directly on both and belongs in a separate control document.

## Table of Contents

1. [First-Order ODEs](#1-first-order-odes)
   - 1.1 [What an ODE Is](#11-what-an-ode-is)
   - 1.2 [Separable Equations](#12-separable-equations)
   - 1.3 [Linear First-Order ODEs & the Time Constant](#13-linear-first-order-odes--the-time-constant)
2. [Second-Order Linear ODEs](#2-second-order-linear-odes)
   - 2.1 [The Mass–Spring–Damper](#21-the-massspringdamper)
   - 2.2 [The Characteristic Equation](#22-the-characteristic-equation)
   - 2.3 [Damping Ratio, Natural Frequency & Step Response](#23-damping-ratio-natural-frequency--step-response)
   - 2.4 [Forced Response](#24-forced-response)
3. [The Laplace Transform](#3-the-laplace-transform)
   - 3.1 [Definition & Basic Pairs](#31-definition--basic-pairs)
   - 3.2 [Key Properties](#32-key-properties)
   - 3.3 [Partial Fractions & the Inverse Transform](#33-partial-fractions--the-inverse-transform)
   - 3.4 [Solving ODEs End to End](#34-solving-odes-end-to-end)
   - 3.5 [Transfer Functions, Impulse Response & Poles](#35-transfer-functions-impulse-response--poles)
   - 3.6 [Systems of ODEs and the Matrix Exponential](#36-systems-of-odes-and-the-matrix-exponential)
4. [Systems of Linear ODEs](#4-systems-of-linear-odes)
   - 4.1 [Reducing Higher-Order ODEs to First-Order Systems](#41-reducing-higher-order-odes-to-first-order-systems)
   - 4.2 [The Free Response & Modes](#42-the-free-response--modes)
   - 4.3 [Phase Portraits in 2D](#43-phase-portraits-in-2d)
5. [State-Space Representation](#5-state-space-representation)
   - 5.1 [The Standard Form](#51-the-standard-form)
   - 5.2 [Solution with an Input](#52-solution-with-an-input)
   - 5.3 [Modeling Examples](#53-modeling-examples)
   - 5.4 [Transfer Functions & Change of Coordinates](#54-transfer-functions--change-of-coordinates)
6. [Equilibria & Linearization](#6-equilibria--linearization)
   - 6.1 [Equilibrium Points](#61-equilibrium-points)
   - 6.2 [Linearization via Jacobians](#62-linearization-via-jacobians)
   - 6.3 [Example: The Pendulum, Hanging & Inverted](#63-example-the-pendulum-hanging--inverted)
   - 6.4 [When Linearization Can Be Trusted](#64-when-linearization-can-be-trusted)
7. [Stability](#7-stability)
   - 7.1 [Definitions](#71-definitions)
   - 7.2 [The Eigenvalue Test](#72-the-eigenvalue-test)
   - 7.3 [Lyapunov's Direct Method](#73-lyapunovs-direct-method)
8. [Discretization](#8-discretization)
   - 8.1 [Why Discretize](#81-why-discretize)
   - 8.2 [Forward & Backward Euler](#82-forward--backward-euler)
   - 8.3 [Exact Discretization (Zero-Order Hold)](#83-exact-discretization-zero-order-hold)
   - 8.4 [Computing A_d and B_d in Practice](#84-computing-a_d-and-b_d-in-practice)
   - 8.5 [Choosing the Sample Time](#85-choosing-the-sample-time)
   - 8.6 [Numerical Integrators for Simulation](#86-numerical-integrators-for-simulation)
9. [Robotics Applications](#9-robotics-applications)
   - 9.1 [Manipulator Dynamics in State-Space Form](#91-manipulator-dynamics-in-state-space-form)
   - 9.2 [The Simulation Loop](#92-the-simulation-loop)
   - 9.3 [Kalman Filter — the Prediction Step](#93-kalman-filter--the-prediction-step)
   - 9.4 [MPC — the Prediction Model](#94-mpc--the-prediction-model)

---

## 1. First-Order ODEs

### 1.1 What an ODE Is

An **ordinary differential equation** relates an unknown function $x(t)$ to its derivatives with respect to one variable (time):

$$\dot{x} = f(x, t), \qquad \dot{x} \equiv \frac{dx}{dt}$$

| Term | Meaning | Example |
|------|---------|---------|
| **Order** | Highest derivative that appears | $m\ddot{x} + c\dot{x} + kx = 0$ is 2nd order |
| **Linear** | $x$ and its derivatives appear only to the first power, not multiplied together | $\dot{x} = -3x + \sin t$ ✓; $\dot{x} = x^2$ ✗; $\ddot\theta = -\sin\theta$ ✗ |
| **Autonomous** | $f$ does not depend on $t$ explicitly | $\dot{x} = -x$ ✓; $\dot{x} = -x + \cos t$ ✗ |
| **Initial value problem (IVP)** | ODE + starting state $x(0) = x_0$ | Pins down one solution out of a family |

**Existence & uniqueness (Picard–Lindelöf):** if $f$ is continuous and **Lipschitz** in $x$ (true whenever $\partial f/\partial x$ is bounded), the IVP has exactly one solution, at least for a short time. This is why a simulator given the same state and input always produces the same trajectory.

"At least for a short time" matters. $\dot{x} = x^2$, $x(0) = 1$ has solution $x(t) = 1/(1 - t)$, which **blows up at $t = 1$** (finite escape time). Linear ODEs never do this — their solutions exist for all time.

### 1.2 Separable Equations

If the right-hand side factors as $\dot{x} = g(x)\,h(t)$, move all $x$ to one side and integrate:

$$\int \frac{dx}{g(x)} = \int h(t)\, dt$$

**Example — exponential decay** $\dot{x} = -kx$:

$$\int \frac{dx}{x} = -\int k\, dt \quad\Rightarrow\quad \ln|x| = -kt + C \quad\Rightarrow\quad x(t) = x_0 e^{-kt}$$

This one solution underlies everything else in the document: **linear systems are sums of exponentials** $e^{\lambda t}$, and $\lambda$ decides whether each piece decays ($\lambda < 0$) or grows ($\lambda > 0$).

**Example — the blow-up above** $\dot{x} = x^2$: $\int x^{-2}\,dx = \int dt \Rightarrow -1/x = t + C$, and $x(0) = 1$ gives $C = -1$, so $x = 1/(1 - t)$.

### 1.3 Linear First-Order ODEs & the Time Constant

General form: $\dot{x} + a\,x = b(t)$. Multiply by the **integrating factor** $e^{at}$; the left side becomes an exact derivative:

$$\frac{d}{dt}\big(e^{at}x\big) = e^{at}\,b(t) \quad\Rightarrow\quad x(t) = \underbrace{e^{-at}x_0}_{\text{free response}} + \underbrace{\int_0^t e^{-a(t-s)}\, b(s)\, ds}_{\text{forced response}}$$

> **Key insight:** the solution is the decaying initial condition **plus** a weighted memory of past inputs, where an input applied at time $s$ has faded by $e^{-a(t-s)}$ by time $t$. The matrix version of this formula (§5.2) is the solution of every linear state-space model.

**Time-constant form.** Engineers write first-order systems as

$$\tau\,\dot{x} + x = K\,u(t)$$

| Parameter | Meaning |
|-----------|---------|
| $\tau$ (time constant) | How fast the system responds; pole at $s = -1/\tau$ |
| $K$ (DC gain) | Final value per unit of constant input |

**Step response** ($u = 1$ for $t \geq 0$, $x_0 = 0$):

$$x(t) = K\big(1 - e^{-t/\tau}\big)$$

| Time | $t = \tau$ | $2\tau$ | $3\tau$ | $4\tau$ | $5\tau$ |
|------|-----------|---------|---------|---------|---------|
| % of final value | 63.2% | 86.5% | 95.0% | 98.2% | 99.3% |

**Example — motor speed with friction.** $J\dot\omega = -b\,\omega + \tau_m$ with inertia $J = 0.01\ \text{kg}\cdot\text{m}^2$, viscous friction $b = 0.1\ \text{N}\cdot\text{m}\cdot\text{s}$. Dividing by $b$: $\tfrac{J}{b}\dot\omega + \omega = \tfrac{1}{b}\tau_m$, so $\tau = J/b = 0.1\ \text{s}$ and $K = 1/b = 10\ \text{rad/s per N}\cdot\text{m}$. A constant $0.5\ \text{N}\cdot\text{m}$ torque drives the motor to $5\ \text{rad/s}$, reaching 95% of it after $0.3\ \text{s}$.

---

## 2. Second-Order Linear ODEs

### 2.1 The Mass–Spring–Damper

The canonical second-order system — and a surprisingly good model of a single robot joint under a PD controller:

$$m\ddot{x} + c\,\dot{x} + k\,x = F(t)$$

| Symbol | Meaning | Robot-joint analogue |
|--------|---------|---------------------|
| $m$ | Mass | Link inertia |
| $c$ | Damping coefficient | Friction + derivative gain $K_d$ |
| $k$ | Spring stiffness | Proportional gain $K_p$ |
| $F(t)$ | External force | Feedforward torque / disturbance |

Dividing by $m$ gives the **standard form**:

$$\ddot{x} + 2\zeta\omega_n\,\dot{x} + \omega_n^2\, x = \frac{F(t)}{m}, \qquad \omega_n = \sqrt{\frac{k}{m}}, \qquad \zeta = \frac{c}{2\sqrt{km}}$$

### 2.2 The Characteristic Equation

Guess $x = e^{st}$ for the unforced equation. Each derivative multiplies by $s$, so:

$$m s^2 + c s + k = 0 \quad\Rightarrow\quad s = -\zeta\omega_n \pm \omega_n\sqrt{\zeta^2 - 1}$$

The two roots $s_1, s_2$ (the **poles**) determine the shape of every solution:

| Case | Roots | General solution |
|------|-------|-----------------|
| **Overdamped** $\zeta > 1$ | Two distinct real, negative | $c_1 e^{s_1 t} + c_2 e^{s_2 t}$ — sluggish, no oscillation |
| **Critically damped** $\zeta = 1$ | Repeated real $s = -\omega_n$ | $(c_1 + c_2 t)\, e^{-\omega_n t}$ — fastest without overshoot |
| **Underdamped** $0 < \zeta < 1$ | Complex pair $-\zeta\omega_n \pm i\omega_d$ | $e^{-\zeta\omega_n t}(c_1\cos\omega_d t + c_2\sin\omega_d t)$ — decaying oscillation |
| **Undamped** $\zeta = 0$ | Pure imaginary $\pm i\omega_n$ | $c_1\cos\omega_n t + c_2\sin\omega_n t$ — oscillates forever |

with the **damped frequency** $\omega_d = \omega_n\sqrt{1 - \zeta^2}$. Complex roots become sines and cosines via Euler's formula $e^{(\sigma + i\omega)t} = e^{\sigma t}(\cos\omega t + i\sin\omega t)$.

**Worked example.** $m = 1$, $c = 2$, $k = 5$, with $x(0) = 1$, $\dot{x}(0) = 0$.

$$s^2 + 2s + 5 = 0 \quad\Rightarrow\quad s = -1 \pm 2i, \qquad \omega_n = \sqrt5 \approx 2.24, \quad \zeta = \tfrac{1}{\sqrt5} \approx 0.447, \quad \omega_d = 2$$

$$x(t) = e^{-t}\big(c_1\cos 2t + c_2\sin 2t\big)$$

$x(0) = 1 \Rightarrow c_1 = 1$. $\dot{x}(0) = -c_1 + 2c_2 = 0 \Rightarrow c_2 = \tfrac12$. So

$$x(t) = e^{-t}\big(\cos 2t + \tfrac12\sin 2t\big)$$

The envelope $e^{-t}$ comes from the **real part** of the poles; the oscillation $\cos 2t$ from the **imaginary part**.

### 2.3 Damping Ratio, Natural Frequency & Step Response

For an underdamped system responding to a unit step, the classic performance measures depend only on $\zeta$ and $\omega_n$:

| Metric | Formula | Depends on |
|--------|---------|-----------|
| Peak time | $t_p = \pi / \omega_d$ | $\omega_n$, $\zeta$ |
| Percent overshoot | $M_p = \exp\!\big(-\zeta\pi / \sqrt{1 - \zeta^2}\big)$ | $\zeta$ only |
| 2% settling time | $t_s \approx 4 / (\zeta\omega_n)$ | Real part of poles |

| $\zeta$ | 0.1 | 0.3 | 0.5 | 0.707 | 1.0 |
|---------|-----|-----|-----|-------|-----|
| Overshoot | 72.9% | 37.2% | 16.3% | 4.3% | 0% |

**For the worked example** ($\zeta = 1/\sqrt5$, $\omega_n = \sqrt5$, $\omega_d = 2$): $\zeta/\sqrt{1 - \zeta^2} = \tfrac12$, so $M_p = e^{-\pi/2} \approx 20.8\%$, $t_p = \pi/2 \approx 1.57\ \text{s}$, $t_s \approx 4/1 = 4\ \text{s}$.

> **Design reading:** $\zeta$ sets the **shape** (how much overshoot), $\omega_n$ sets the **speed** (how fast it happens). $\zeta \approx 0.7$ is a common target for robot joints: fast, with only a few percent overshoot.

**Pole geometry.** In the complex plane, the poles sit at distance $\omega_n$ from the origin, at angle $\arccos\zeta$ from the negative real axis. Moving poles **left** = faster decay; moving them **away from the real axis** = more oscillation.

### 2.4 Forced Response

For a linear ODE, **general solution = homogeneous solution + one particular solution**:

$$x(t) = x_h(t) + x_p(t)$$

The homogeneous part (§2.2) carries the initial conditions and dies out if the system is stable; the particular part is the steady response to the input.

| Input $F(t)$ | Try $x_p$ | Result for $m\ddot{x} + c\dot{x} + kx = F$ |
|--------------|-----------|---------------------------------------------|
| Constant $F_0$ | Constant $X$ | $X = F_0 / k$ — the spring's static deflection |
| $F_0\cos\omega t$ | $X\cos(\omega t - \phi)$ | $X = \dfrac{F_0}{\sqrt{(k - m\omega^2)^2 + (c\omega)^2}}$ |

**Resonance.** When $\zeta$ is small, the sinusoidal amplitude $X$ peaks sharply near $\omega \approx \omega_n$ (exactly at $\omega_n\sqrt{1 - 2\zeta^2}$ for $\zeta < 1/\sqrt2$). At $\omega = \omega_n$ the stiffness and inertia terms cancel ($k - m\omega_n^2 = 0$) and only damping limits the motion: $X = F_0/(c\,\omega_n)$.

**In robotics:** a lightly damped flexible link or gearbox excited near its natural frequency vibrates badly — this is why trajectories are smoothed (bounded jerk) and controller bandwidth is kept well below structural resonances.

---

## 3. The Laplace Transform

The Laplace transform turns a linear ODE with constant coefficients into an **algebraic** equation in a new variable $s$. Derivatives become multiplication by $s$, initial conditions enter automatically, and step, impulse, or switched inputs are no harder than smooth ones. Solving becomes a mechanical pipeline:

$$\text{ODE in } t \xrightarrow{\ \mathcal{L}\ } \text{algebra in } s \xrightarrow{\ \text{solve for } X(s)\ } \text{partial fractions} \xrightarrow{\ \mathcal{L}^{-1}\ } x(t)$$

Everything in §2 can be done without it — but the Laplace transform is faster for forced problems and is the native language of transfer functions, poles, and control design.

### 3.1 Definition & Basic Pairs

For a function $f(t)$ defined for $t \geq 0$:

$$F(s) = \mathcal{L}\{f(t)\} = \int_0^\infty f(t)\, e^{-st}\, dt, \qquad s \in \mathbb{C}$$

The integral converges when $\operatorname{Re}(s)$ is large enough to beat the growth of $f$ (the **region of convergence**). For every signal in this document that holds for some half-plane $\operatorname{Re}(s) > \sigma_0$, and in practice you never need to compute it — just use the table.

**Computing one pair** — $f(t) = e^{at}$:

$$F(s) = \int_0^\infty e^{at}e^{-st}\, dt = \int_0^\infty e^{-(s-a)t}\, dt = \left[\frac{-e^{-(s-a)t}}{s - a}\right]_0^\infty = \frac{1}{s - a} \qquad (\operatorname{Re}\,s > a)$$

Setting $a = 0$ gives the unit step: $\mathcal{L}\{1\} = 1/s$.

**Transform table** ($u(t)$ = unit step, $\delta(t)$ = unit impulse):

| $f(t)$, $t \geq 0$ | $F(s)$ | Pole location |
|-------------------|--------|---------------|
| $\delta(t)$ | $1$ | — |
| $u(t)$ (or $1$) | $\dfrac{1}{s}$ | $0$ |
| $t$ | $\dfrac{1}{s^2}$ | $0$ (double) |
| $t^n$ | $\dfrac{n!}{s^{n+1}}$ | $0$ ($n+1$-fold) |
| $e^{at}$ | $\dfrac{1}{s - a}$ | $a$ |
| $t\,e^{at}$ | $\dfrac{1}{(s - a)^2}$ | $a$ (double) |
| $\sin\omega t$ | $\dfrac{\omega}{s^2 + \omega^2}$ | $\pm i\omega$ |
| $\cos\omega t$ | $\dfrac{s}{s^2 + \omega^2}$ | $\pm i\omega$ |
| $e^{at}\sin\omega t$ | $\dfrac{\omega}{(s - a)^2 + \omega^2}$ | $a \pm i\omega$ |
| $e^{at}\cos\omega t$ | $\dfrac{s - a}{(s - a)^2 + \omega^2}$ | $a \pm i\omega$ |
| $u(t - T)$ (step delayed by $T$) | $\dfrac{e^{-Ts}}{s}$ | $0$ |

> **Read the table backwards:** a pole at $s = a$ in $F(s)$ means a term $e^{at}$ in $f(t)$. Real part of the pole → growth/decay rate; imaginary part → oscillation frequency. This is the same pole–behavior link as §2.2 and §7.2.

### 3.2 Key Properties

| Property | Time domain | $s$-domain |
|----------|-------------|-----------|
| Linearity | $a f(t) + b g(t)$ | $aF(s) + bG(s)$ |
| **First derivative** | $\dot{f}(t)$ | $sF(s) - f(0)$ |
| **Second derivative** | $\ddot{f}(t)$ | $s^2F(s) - s f(0) - \dot{f}(0)$ |
| Integral | $\int_0^t f(\tau)\, d\tau$ | $\dfrac{F(s)}{s}$ |
| $s$-shift (damping) | $e^{at}f(t)$ | $F(s - a)$ |
| $t$-shift (delay) | $f(t - T)\,u(t - T)$ | $e^{-Ts}F(s)$ |
| **Convolution** | $(f * g)(t) = \int_0^t f(\tau)\,g(t - \tau)\, d\tau$ | $F(s)\,G(s)$ |
| Final value | $\lim_{t\to\infty} f(t)$ | $\lim_{s\to 0} sF(s)$ — only if all poles of $sF(s)$ have $\operatorname{Re} < 0$ |
| Initial value | $f(0^+)$ | $\lim_{s\to\infty} sF(s)$ |

**Why the derivative rule holds** (integration by parts):

$$\mathcal{L}\{\dot{f}\} = \int_0^\infty \dot{f}\,e^{-st}\, dt = \Big[f\,e^{-st}\Big]_0^\infty + s\int_0^\infty f\,e^{-st}\, dt = -f(0) + sF(s)$$

Apply it twice for $\ddot{f}$. This is the property that makes the method work: **differentiation becomes multiplication by $s$, and the initial conditions appear as constants.**

**Final value theorem warning:** for $F(s) = 1/(s - 1)$, the formula gives $\lim_{s\to 0} s/(s - 1) = 0$, but $f(t) = e^t \to \infty$. Check that the system is stable before using it.

### 3.3 Partial Fractions & the Inverse Transform

The solution $X(s)$ almost always comes out as a ratio of polynomials. Split it into table entries, one per pole:

**Case 1 — distinct real poles.** $X(s) = \dfrac{s + 3}{(s + 1)(s + 2)} = \dfrac{A}{s + 1} + \dfrac{B}{s + 2}$.

**Cover-up rule:** multiply by one factor and set $s$ to its root.

$$A = \frac{s + 3}{s + 2}\bigg|_{s = -1} = \frac{2}{1} = 2, \qquad B = \frac{s + 3}{s + 1}\bigg|_{s = -2} = \frac{1}{-1} = -1$$

$$x(t) = 2e^{-t} - e^{-2t}$$

**Case 2 — repeated pole.** Include every power up to the multiplicity:

$$\frac{s + 3}{(s + 1)^2} = \frac{A}{s + 1} + \frac{B}{(s + 1)^2}, \qquad s + 3 = A(s + 1) + B \ \Rightarrow\ A = 1,\ B = 2$$

$$x(t) = e^{-t} + 2t\,e^{-t}$$

The repeated pole produces the $t\,e^{at}$ term — the time-domain signature of critical damping (§2.2).

**Case 3 — complex poles.** Do not split into complex factors; **complete the square** and match the sine/cosine rows of the table:

$$\frac{s + 2}{s^2 + 2s + 5} = \frac{s + 2}{(s + 1)^2 + 2^2} = \frac{(s + 1)}{(s + 1)^2 + 2^2} + \frac{1}{2}\cdot\frac{2}{(s + 1)^2 + 2^2}$$

$$x(t) = e^{-t}\cos 2t + \tfrac12 e^{-t}\sin 2t$$

**Mixed case:** a real pole and a complex pair use $\dfrac{A}{s - a} + \dfrac{Bs + C}{(s - \alpha)^2 + \omega^2}$; match coefficients of powers of $s$ (worked in §3.4).

### 3.4 Solving ODEs End to End

**Example 1 — free response (redo of §2.2).** $\ddot{x} + 2\dot{x} + 5x = 0$, $x(0) = 1$, $\dot{x}(0) = 0$.

Transform term by term with the derivative rules:

$$\underbrace{s^2X - s\cdot 1 - 0}_{\mathcal{L}\{\ddot{x}\}} + 2\underbrace{(sX - 1)}_{\mathcal{L}\{\dot{x}\}} + 5X = 0 \quad\Rightarrow\quad (s^2 + 2s + 5)X = s + 2 \quad\Rightarrow\quad X(s) = \frac{s + 2}{s^2 + 2s + 5}$$

By Case 3 above: $x(t) = e^{-t}\big(\cos 2t + \tfrac12\sin 2t\big)$ — identical to §2.2, with no constants to fit afterwards. ✓

**Example 2 — step response.** Same system at rest, hit with a unit step force: $\ddot{x} + 2\dot{x} + 5x = u(t)$, $x(0) = \dot{x}(0) = 0$.

$$(s^2 + 2s + 5)X = \frac{1}{s} \quad\Rightarrow\quad X(s) = \frac{1}{s\,(s^2 + 2s + 5)} = \frac{A}{s} + \frac{Bs + C}{s^2 + 2s + 5}$$

Cover-up gives $A = 1/5$. Matching coefficients of $1 = A(s^2 + 2s + 5) + (Bs + C)s$: $s^2$: $A + B = 0 \Rightarrow B = -\tfrac15$; $s^1$: $2A + C = 0 \Rightarrow C = -\tfrac25$.

$$X(s) = \frac15\left[\frac1s - \frac{s + 2}{s^2 + 2s + 5}\right] \quad\Rightarrow\quad x(t) = \frac15\Big[1 - e^{-t}\big(\cos 2t + \tfrac12\sin 2t\big)\Big]$$

Checks:
- Final value theorem: $\lim_{s\to 0} sX(s) = \tfrac15$ = static deflection $F_0/k$ (§2.4). ✓
- Peak overshoot of the bracket is $e^{-\pi/2} \approx 20.8\%$ — matching the formula in §2.3. ✓

**Example 3 — impulse response.** A sharp hammer tap, $\ddot{x} + 2\dot{x} + 5x = \delta(t)$, at rest:

$$X(s) = \frac{1}{s^2 + 2s + 5} = \frac12\cdot\frac{2}{(s + 1)^2 + 2^2} \quad\Rightarrow\quad x(t) = \tfrac12 e^{-t}\sin 2t$$

It equals the derivative of the step response (differentiate Example 2 to check) — because $\delta$ is the derivative of $u$, and the system is linear.

**Example 4 — switched input.** A force pulse of height 1 lasting $T_0$ seconds is $u(t) - u(t - T_0)$, with transform $\dfrac{1 - e^{-T_0 s}}{s}$. By linearity and the $t$-shift rule the response is

$$x(t) = x_{\text{step}}(t) - x_{\text{step}}(t - T_0)\,u(t - T_0)$$

— no need to solve the ODE piecewise and match conditions at $t = T_0$.

### 3.5 Transfer Functions, Impulse Response & Poles

With **zero initial conditions**, every linear constant-coefficient ODE becomes $X(s) = G(s)\,U(s)$. For the mass–spring–damper:

$$m s^2 X + c s X + k X = U \quad\Rightarrow\quad G(s) = \frac{X(s)}{U(s)} = \frac{1}{ms^2 + cs + k}$$

| Object | Meaning |
|--------|---------|
| $G(s)$ | **Transfer function** — the system itself, independent of the input |
| $g(t) = \mathcal{L}^{-1}\{G(s)\}$ | **Impulse response** — output when $U(s) = 1$, i.e. $u = \delta$ |
| $x(t) = (g * u)(t)$ | Output for any input — the convolution theorem turns $GU$ into this integral; it is the forced-response term of §5.2 |
| Poles of $G$ | Roots of the denominator = roots of the characteristic equation (§2.2) — they set the transient shape |
| Zeros of $G$ | Roots of the numerator — they shape how strongly each mode is excited |

**Poles → time response at a glance:**

| Pole location in the $s$-plane | Term in $g(t)$ |
|--------------------------------|---------------|
| Negative real, $s = -a$ | $e^{-at}$ — decays; farther left = faster |
| Complex pair, $-\sigma \pm i\omega$, $\sigma > 0$ | $e^{-\sigma t}\sin(\omega t + \phi)$ — decaying oscillation |
| On the imaginary axis, $\pm i\omega$ | Sustained oscillation |
| At the origin, $s = 0$ | Constant (an integrator) |
| Any pole with $\operatorname{Re} > 0$ | Exponential growth — unstable |

Evaluating $G$ on the imaginary axis, $G(i\omega)$, gives the steady-state response to a sinusoid of frequency $\omega$ — the frequency response, developed in [signals_systems.md](signals_systems.md) §4.

### 3.6 Systems of ODEs and the Matrix Exponential

Transform the state equation $\dot{\mathbf{x}} = A\mathbf{x} + B\mathbf{u}$ entry by entry:

$$s\mathbf{X}(s) - \mathbf{x}_0 = A\mathbf{X}(s) + B\mathbf{U}(s) \quad\Rightarrow\quad \mathbf{X}(s) = (sI - A)^{-1}\mathbf{x}_0 + (sI - A)^{-1}B\,\mathbf{U}(s)$$

Compare with the time-domain solution $\mathbf{x}(t) = e^{At}\mathbf{x}_0 + \int_0^t e^{A(t - \tau)}B\mathbf{u}(\tau)\,d\tau$ (§5.2): term by term,

$$\boxed{\ \mathcal{L}\{e^{At}\} = (sI - A)^{-1}\ }$$

and the convolution integral becomes a product. The matrix $(sI - A)^{-1}$ is called the **resolvent**; its entries have denominator $\det(sI - A)$, so **the poles are the eigenvalues of $A$**.

**Worked example — $e^{At}$ by Laplace.** $A = \begin{bmatrix} 0 & 1 \\ -2 & -3 \end{bmatrix}$ (the system of §4.2).

$$sI - A = \begin{bmatrix} s & -1 \\ 2 & s + 3 \end{bmatrix}, \qquad (sI - A)^{-1} = \frac{1}{(s + 1)(s + 2)}\begin{bmatrix} s + 3 & 1 \\ -2 & s \end{bmatrix}$$

Invert each entry with partial fractions (Case 1):

| Entry | Partial fractions | Inverse |
|-------|------------------|---------|
| $\dfrac{s + 3}{(s+1)(s+2)}$ | $\dfrac{2}{s+1} - \dfrac{1}{s+2}$ | $2e^{-t} - e^{-2t}$ |
| $\dfrac{1}{(s+1)(s+2)}$ | $\dfrac{1}{s+1} - \dfrac{1}{s+2}$ | $e^{-t} - e^{-2t}$ |
| $\dfrac{-2}{(s+1)(s+2)}$ | $\dfrac{-2}{s+1} + \dfrac{2}{s+2}$ | $-2e^{-t} + 2e^{-2t}$ |
| $\dfrac{s}{(s+1)(s+2)}$ | $\dfrac{-1}{s+1} + \dfrac{2}{s+2}$ | $-e^{-t} + 2e^{-2t}$ |

$$e^{At} = \begin{bmatrix} 2e^{-t} - e^{-2t} & e^{-t} - e^{-2t} \\ -2e^{-t} + 2e^{-2t} & -e^{-t} + 2e^{-2t} \end{bmatrix}$$

Checks: at $t = 0$ this is $I$ ✓; its first column is the solution for $\mathbf{x}_0 = (1, 0)$ found by eigenvectors in §4.2 ✓.

With an output $\mathbf{y} = C\mathbf{x} + D\mathbf{u}$ and zero initial state, $\mathbf{Y}(s) = \big[C(sI - A)^{-1}B + D\big]\mathbf{U}(s)$ — the state-space transfer function of §5.4.

| Method for $\mathbf{x}(t)$ | Best when |
|----------------------------|-----------|
| Eigenvectors (§4.2) | You also want the mode shapes; $A$ diagonalizable |
| Laplace / resolvent (§3.6) | Small systems by hand; repeated eigenvalues; forced or switched inputs |
| `expm` / numerical (§8.4) | Anything larger than $3 \times 3$, or real code |

---

## 4. Systems of Linear ODEs

### 4.1 Reducing Higher-Order ODEs to First-Order Systems

**Any** $n$-th order ODE becomes $n$ coupled first-order ODEs by choosing the state as the variable and its first $n-1$ derivatives. For the mass–spring–damper, let $x_1 = x$ (position), $x_2 = \dot{x}$ (velocity):

$$\begin{aligned} \dot{x}_1 &= x_2 \\ \dot{x}_2 &= -\tfrac{k}{m}x_1 - \tfrac{c}{m}x_2 + \tfrac{1}{m}F \end{aligned}
\qquad\Longleftrightarrow\qquad
\frac{d}{dt}\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \underbrace{\begin{bmatrix} 0 & 1 \\ -k/m & -c/m \end{bmatrix}}_{A}\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} + \underbrace{\begin{bmatrix} 0 \\ 1/m \end{bmatrix}}_{B} F$$

**Check:** $\det(sI - A) = s^2 + \tfrac{c}{m}s + \tfrac{k}{m}$ — the characteristic equation of §2.2 divided by $m$. **The eigenvalues of $A$ are exactly the poles.**

**General pattern** — $x^{(n)} + a_{n-1}x^{(n-1)} + \cdots + a_1\dot{x} + a_0 x = u$ becomes the **companion form**:

$$A = \begin{bmatrix} 0 & 1 & & \\ & \ddots & \ddots & \\ & & 0 & 1 \\ -a_0 & -a_1 & \cdots & -a_{n-1} \end{bmatrix}, \qquad B = \begin{bmatrix} 0 \\ \vdots \\ 0 \\ 1 \end{bmatrix}$$

> **Why first-order form:** one equation type, $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u})$, covers every system. All the theory (§5–7) and every numerical integrator (§8) are written for it.

### 4.2 The Free Response & Modes

The vector analogue of $\dot{x} = ax \Rightarrow x = e^{at}x_0$ is

$$\dot{\mathbf{x}} = A\mathbf{x} \quad\Longrightarrow\quad \mathbf{x}(t) = e^{At}\,\mathbf{x}_0$$

with the **matrix exponential** $e^{At} = I + At + \tfrac{(At)^2}{2!} + \cdots$ (definition and properties in [linear_algebra.md](../basics/linear_algebra.md) §14.4).

**Modal solution.** If $A$ has $n$ independent eigenvectors $A\mathbf{v}_i = \lambda_i\mathbf{v}_i$:

$$\mathbf{x}(t) = c_1 e^{\lambda_1 t}\mathbf{v}_1 + \cdots + c_n e^{\lambda_n t}\mathbf{v}_n, \qquad \text{where } \mathbf{x}_0 = c_1\mathbf{v}_1 + \cdots + c_n\mathbf{v}_n$$

Each eigenvector is a **mode**: start on it and you stay on it, scaling by $e^{\lambda_i t}$.

**Worked example.** $A = \begin{bmatrix} 0 & 1 \\ -2 & -3 \end{bmatrix}$ (that is, $\ddot{x} + 3\dot{x} + 2x = 0$), $\mathbf{x}_0 = (1, 0)$.

1. **Eigenvalues:** $\det(A - \lambda I) = \lambda^2 + 3\lambda + 2 = (\lambda + 1)(\lambda + 2) \Rightarrow \lambda_1 = -1,\ \lambda_2 = -2$.
2. **Eigenvectors:** $(A + I)\mathbf{v} = \mathbf{0} \Rightarrow \mathbf{v}_1 = (1, -1)$. $(A + 2I)\mathbf{v} = \mathbf{0} \Rightarrow \mathbf{v}_2 = (1, -2)$.
3. **Coefficients:** $c_1\mathbf{v}_1 + c_2\mathbf{v}_2 = (1, 0) \Rightarrow c_1 + c_2 = 1,\ -c_1 - 2c_2 = 0 \Rightarrow c_1 = 2,\ c_2 = -1$.
4. **Solution:**

$$\mathbf{x}(t) = 2e^{-t}\begin{bmatrix} 1 \\ -1 \end{bmatrix} - e^{-2t}\begin{bmatrix} 1 \\ -2 \end{bmatrix} = \begin{bmatrix} 2e^{-t} - e^{-2t} \\ -2e^{-t} + 2e^{-2t} \end{bmatrix}$$

Check: $\dot{x}_1 = -2e^{-t} + 2e^{-2t} = x_2$ ✓ (velocity is the derivative of position). The $e^{-2t}$ mode dies first; for large $t$ the state slides in along $\mathbf{v}_1$ — the **slow mode dominates**.

**Other eigenvalue structures:**

| Structure | What appears in $\mathbf{x}(t)$ | Example |
|-----------|-------------------------------|---------|
| Complex pair $\sigma \pm i\omega$ | $e^{\sigma t}\cos\omega t$, $e^{\sigma t}\sin\omega t$ | §2.2 underdamped |
| Repeated eigenvalue, too few eigenvectors | $t\,e^{\lambda t}$ (and $t^2 e^{\lambda t}$, … — Jordan blocks) | Critical damping |
| $A$ nilpotent | Polynomial in $t$ — the series stops | Double integrator below |

**Double integrator** $\ddot{x} = 0$ (a mass with no forces): $A = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}$, $A^2 = 0$, so

$$e^{At} = I + At = \begin{bmatrix} 1 & t \\ 0 & 1 \end{bmatrix} \quad\Rightarrow\quad x(t) = x_0 + v_0 t,\ \ v(t) = v_0 \quad\checkmark$$

### 4.3 Phase Portraits in 2D

For $\dot{\mathbf{x}} = A\mathbf{x}$ with $A \in \mathbb{R}^{2 \times 2}$, plotting trajectories in the $(x_1, x_2)$ plane gives the **phase portrait**. Its type depends only on the trace $T = \operatorname{tr}A = \lambda_1 + \lambda_2$ and determinant $\Delta = \det A = \lambda_1\lambda_2$:

| Condition | Eigenvalues | Type | Picture |
|-----------|------------|------|---------|
| $\Delta < 0$ | Real, opposite signs | **Saddle** (unstable) | In along one eigenvector, out along the other |
| $\Delta > 0$, $T^2 > 4\Delta$, $T < 0$ | Real, both negative | **Stable node** | Straight-ish curves into the origin |
| $\Delta > 0$, $T^2 > 4\Delta$, $T > 0$ | Real, both positive | **Unstable node** | Curves out of the origin |
| $\Delta > 0$, $T^2 < 4\Delta$, $T < 0$ | Complex, $\operatorname{Re} < 0$ | **Stable spiral** | Spirals inward |
| $\Delta > 0$, $T^2 < 4\Delta$, $T > 0$ | Complex, $\operatorname{Re} > 0$ | **Unstable spiral** | Spirals outward |
| $\Delta > 0$, $T = 0$ | Pure imaginary | **Center** | Closed ellipses — neither in nor out |

**Quick test:** a 2D linear system is asymptotically stable **iff $T < 0$ and $\Delta > 0$**.

Reading the examples: $A = \begin{bmatrix} 0 & 1 \\ -2 & -3 \end{bmatrix}$ has $T = -3$, $\Delta = 2$, $T^2 = 9 > 8$ → stable node. The §2.2 example ($T = -2$, $\Delta = 5$, $T^2 = 4 < 20$) → stable spiral. An undamped spring ($T = 0$) → center.

---

## 5. State-Space Representation

### 5.1 The Standard Form

A **linear time-invariant (LTI) state-space model**:

$$\boxed{\ \dot{\mathbf{x}} = A\mathbf{x} + B\mathbf{u}, \qquad \mathbf{y} = C\mathbf{x} + D\mathbf{u}\ }$$

| Symbol | Shape | Name | Meaning |
|--------|-------|------|---------|
| $\mathbf{x}$ | $n \times 1$ | State | Everything you need to know now to predict the future (positions, velocities, currents, …) |
| $\mathbf{u}$ | $p \times 1$ | Input | What you command (torques, voltages, thrusts) |
| $\mathbf{y}$ | $q \times 1$ | Output | What you measure (encoder angles, IMU readings) |
| $A$ | $n \times n$ | System matrix | Internal dynamics — its eigenvalues are the poles |
| $B$ | $n \times p$ | Input matrix | How inputs push on the state |
| $C$ | $q \times n$ | Output matrix | Which combinations of state the sensors see |
| $D$ | $q \times p$ | Feedthrough | Direct input-to-output path; $0$ for most physical systems |

> **The state is the system's memory.** Given $\mathbf{x}(t_0)$ and the input $\mathbf{u}$ on $[t_0, t]$, the future is determined — nothing about the past before $t_0$ is needed. Choosing a state means choosing the smallest set of variables with that property. For mechanical systems the natural choice is **(positions, velocities)**.

**Nonlinear form.** Real robots are $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u})$, $\mathbf{y} = \mathbf{h}(\mathbf{x}, \mathbf{u})$. The LTI form is the local approximation obtained in §6.

### 5.2 Solution with an Input

The matrix version of §1.3. Multiply $\dot{\mathbf{x}} - A\mathbf{x} = B\mathbf{u}$ by $e^{-At}$:

$$\frac{d}{dt}\big(e^{-At}\mathbf{x}\big) = e^{-At}\big(\dot{\mathbf{x}} - A\mathbf{x}\big) = e^{-At}B\mathbf{u}$$

Integrate from $0$ to $t$ and multiply by $e^{At}$:

$$\boxed{\ \mathbf{x}(t) = \underbrace{e^{At}\mathbf{x}_0}_{\text{free response}} + \underbrace{\int_0^t e^{A(t - s)}B\,\mathbf{u}(s)\, ds}_{\text{forced response}}\ }$$

This is the **variation-of-constants** (or convolution) formula. Two consequences used constantly:

**Constant input** $\mathbf{u}(t) = \mathbf{u}_0$, $A$ invertible:

$$\mathbf{x}(t) = e^{At}\mathbf{x}_0 + A^{-1}\big(e^{At} - I\big)B\mathbf{u}_0$$

**Steady state** (if $A$ is stable, $e^{At} \to 0$):

$$\mathbf{x}_{ss} = -A^{-1}B\mathbf{u}_0 \qquad (\text{equivalently, solve } \mathbf{0} = A\mathbf{x}_{ss} + B\mathbf{u}_0)$$

Mass–spring–damper with constant force $F_0$: $\mathbf{0} = A\mathbf{x}_{ss} + BF_0$ gives $x_2 = 0$ and $-\tfrac{k}{m}x_1 + \tfrac{F_0}{m} = 0$, so $\mathbf{x}_{ss} = (F_0/k,\ 0)$ — matching §2.4. ✓

**Superposition.** Because the solution is linear in $(\mathbf{x}_0, \mathbf{u})$, responses add: response to ($\mathbf{x}_0$, $\mathbf{u}_1 + \mathbf{u}_2$) = free response + forced response to $\mathbf{u}_1$ + forced response to $\mathbf{u}_2$. Nonlinear systems lose this property.

### 5.3 Modeling Examples

**Example 1 — Mass–spring–damper** (from §4.1), measuring position:

$$A = \begin{bmatrix} 0 & 1 \\ -k/m & -c/m \end{bmatrix}, \quad B = \begin{bmatrix} 0 \\ 1/m \end{bmatrix}, \quad C = \begin{bmatrix} 1 & 0 \end{bmatrix}, \quad D = 0$$

**Example 2 — DC motor driving a joint** (voltage input, encoder output). Motor torque is $K i$, back-EMF is $K\omega$; with small winding inductance the current is $i = (V - K\omega)/R$. Newton's law for the rotor:

$$J\dot\omega = -b\,\omega + K i = -\Big(b + \frac{K^2}{R}\Big)\omega + \frac{K}{R}V$$

With state $\mathbf{x} = (\theta, \omega)$, input $u = V$, output $y = \theta$:

$$A = \begin{bmatrix} 0 & 1 \\ 0 & -\dfrac{b + K^2/R}{J} \end{bmatrix}, \quad B = \begin{bmatrix} 0 \\ \dfrac{K}{RJ} \end{bmatrix}, \quad C = \begin{bmatrix} 1 & 0 \end{bmatrix}$$

Eigenvalues: $0$ and $-(b + K^2/R)/J$. The $0$ eigenvalue says **position integrates velocity** — hold a constant voltage and the angle grows without bound. The back-EMF term $K^2/R$ acts as **electrical damping** on top of mechanical friction.

**Example 3 — Planar point mass (double integrator)**, the standard model for a drone's or mobile base's translational motion with force input:

$$\mathbf{x} = \begin{bmatrix} p_x \\ p_y \\ v_x \\ v_y \end{bmatrix}, \quad A = \begin{bmatrix} 0 & I_2 \\ 0 & 0 \end{bmatrix}, \quad B = \begin{bmatrix} 0 \\ \tfrac1m I_2 \end{bmatrix}$$

All four eigenvalues are $0$ — not asymptotically stable. Without feedback a pushed mass drifts forever.

### 5.4 Transfer Functions & Change of Coordinates

**Transfer function.** Taking Laplace transforms with zero initial state (§3.6):

$$G(s) = \frac{Y(s)}{U(s)} = C(sI - A)^{-1}B + D$$

Mass–spring–damper: $sI - A = \begin{bmatrix} s & -1 \\ k/m & s + c/m \end{bmatrix}$, $(sI - A)^{-1} = \dfrac{1}{s^2 + \frac{c}{m}s + \frac{k}{m}}\begin{bmatrix} s + c/m & 1 \\ -k/m & s \end{bmatrix}$, so

$$G(s) = \begin{bmatrix} 1 & 0 \end{bmatrix}(sI - A)^{-1}\begin{bmatrix} 0 \\ 1/m \end{bmatrix} = \frac{1}{ms^2 + cs + k}$$

The denominator is $\det(sI - A)$, so **poles of $G$ = eigenvalues of $A$** (barring cancellations).

| | Transfer function $G(s)$ | State space $(A, B, C, D)$ |
|---|---|---|
| Describes | Input → output only | Full internal state |
| Multi-input / multi-output | Awkward (matrix of ratios) | Natural |
| Nonzero initial conditions | Not directly | Built in |
| Used for | Frequency response, classical (PID) tuning | Simulation, estimation, MPC, LQR |

**Change of coordinates.** The state is not unique. For any invertible $T$, the new state $\mathbf{z} = T\mathbf{x}$ obeys

$$\dot{\mathbf{z}} = TAT^{-1}\mathbf{z} + TB\,\mathbf{u}, \qquad \mathbf{y} = CT^{-1}\mathbf{z} + D\mathbf{u}$$

$TAT^{-1}$ is **similar** to $A$ ([linear_algebra.md](../basics/linear_algebra.md) §3.4) — same eigenvalues, same transfer function, same physics. Choosing $T^{-1}$ = eigenvectors of $A$ gives **modal coordinates** in which $A$ is diagonal and every mode evolves independently.

---

## 6. Equilibria & Linearization

### 6.1 Equilibrium Points

For $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u})$, an **equilibrium** (operating point) $(\mathbf{x}^*, \mathbf{u}^*)$ is where nothing changes:

$$\mathbf{f}(\mathbf{x}^*, \mathbf{u}^*) = \mathbf{0}$$

For mechanical systems with $\mathbf{x} = (\mathbf{q}, \dot{\mathbf{q}})$ this always means **zero velocity**, and the input $\mathbf{u}^*$ exactly balances the static forces (gravity, springs). A robot arm can be held at almost any pose $\mathbf{q}^*$ — the required holding torque is $\boldsymbol\tau^* = \mathbf{g}(\mathbf{q}^*)$ (§9.1).

Nonlinear systems can have **several** equilibria, each with its own local behavior. A linear system $\dot{\mathbf{x}} = A\mathbf{x}$ has only the origin (or a whole subspace if $A$ is singular).

### 6.2 Linearization via Jacobians

Write the state and input as deviations from the operating point: $\delta\mathbf{x} = \mathbf{x} - \mathbf{x}^*$, $\delta\mathbf{u} = \mathbf{u} - \mathbf{u}^*$. A first-order Taylor expansion ([multivariable_calculus.md](../basics/multivariable_calculus.md) §6.2) gives

$$\dot{\delta\mathbf{x}} = \mathbf{f}(\mathbf{x}^* + \delta\mathbf{x}, \mathbf{u}^* + \delta\mathbf{u}) \approx \underbrace{\mathbf{f}(\mathbf{x}^*, \mathbf{u}^*)}_{=\,\mathbf{0}} + \frac{\partial\mathbf{f}}{\partial\mathbf{x}}\bigg|_*\delta\mathbf{x} + \frac{\partial\mathbf{f}}{\partial\mathbf{u}}\bigg|_*\delta\mathbf{u}$$

$$\boxed{\ \dot{\delta\mathbf{x}} \approx A\,\delta\mathbf{x} + B\,\delta\mathbf{u}, \qquad A = \frac{\partial\mathbf{f}}{\partial\mathbf{x}}\bigg|_{(\mathbf{x}^*, \mathbf{u}^*)} \in \mathbb{R}^{n \times n}, \quad B = \frac{\partial\mathbf{f}}{\partial\mathbf{u}}\bigg|_{(\mathbf{x}^*, \mathbf{u}^*)} \in \mathbb{R}^{n \times p}\ }$$

Both are Jacobians in numerator layout (row $i$ = gradient of $f_i$; [multivariable_calculus.md](../basics/multivariable_calculus.md) §6.5). Similarly $C = \partial\mathbf{h}/\partial\mathbf{x}|_*$ for the output.

**Recipe:**

1. Write the model as first-order $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u})$ (§4.1).
2. Solve $\mathbf{f}(\mathbf{x}^*, \mathbf{u}^*) = \mathbf{0}$ for the operating point.
3. Differentiate $\mathbf{f}$ symbolically to get the Jacobians.
4. **Evaluate at the operating point** — $A$ and $B$ are constant matrices.

### 6.3 Example: The Pendulum, Hanging & Inverted

A rigid pendulum (mass $m$, length $l$, viscous friction $b$, motor torque $\tau$), angle $\theta$ measured from straight down:

$$ml^2\ddot\theta = -mgl\sin\theta - b\dot\theta + \tau$$

**Step 1 — state form.** $\mathbf{x} = (x_1, x_2) = (\theta, \dot\theta)$, $u = \tau$. Writing $\beta = b/(ml^2)$:

$$\mathbf{f}(\mathbf{x}, u) = \begin{bmatrix} x_2 \\ -\dfrac{g}{l}\sin x_1 - \beta x_2 + \dfrac{u}{ml^2} \end{bmatrix}$$

**Step 2 — equilibria** with $u^* = 0$: $x_2 = 0$ and $\sin x_1 = 0$, so $\theta^* = 0$ (hanging) or $\theta^* = \pi$ (inverted).

**Step 3 — Jacobians:**

$$A(\mathbf{x}) = \frac{\partial\mathbf{f}}{\partial\mathbf{x}} = \begin{bmatrix} 0 & 1 \\ -\dfrac{g}{l}\cos x_1 & -\beta \end{bmatrix}, \qquad B = \frac{\partial\mathbf{f}}{\partial u} = \begin{bmatrix} 0 \\ \dfrac{1}{ml^2} \end{bmatrix}$$

**Step 4 — evaluate:**

| Operating point | $A$ | $\Delta = \det A$ | Type (§4.3) |
|-----------------|-----|-------------------|-------------|
| Hanging, $\theta^* = 0$ | $\begin{bmatrix} 0 & 1 \\ -g/l & -\beta \end{bmatrix}$ | $g/l > 0$, $T = -\beta < 0$ | **Stable** (spiral or node) |
| Inverted, $\theta^* = \pi$ | $\begin{bmatrix} 0 & 1 \\ +g/l & -\beta \end{bmatrix}$ | $-g/l < 0$ | **Saddle — unstable** |

The only change is the sign of $\cos\theta^*$: gravity is a **restoring spring** when hanging and an **anti-spring** when inverted.

**Concrete numbers** (inverted, $l = 1\ \text{m}$, $g = 9.81$, $\beta = 0$): eigenvalues $\pm\sqrt{9.81} \approx \pm 3.13\ \text{s}^{-1}$. A small tilt grows like $e^{3.13t}$ — it **doubles every $\ln 2 / 3.13 \approx 0.22\ \text{s}$**. A balancing controller must react well within that time, which sets a lower bound on its sample rate (§8.5).

**Non-zero operating points.** To hold the arm horizontal ($\theta^* = \pi/2$) the motor must supply $u^* = mgl$. There $\cos\theta^* = 0$, so $A = \begin{bmatrix} 0 & 1 \\ 0 & -\beta \end{bmatrix}$ — gravity's stiffness vanishes and the joint behaves locally like a damped free mass.

### 6.4 When Linearization Can Be Trusted

**Hartman–Grobman theorem:** if no eigenvalue of $A$ has zero real part (a **hyperbolic** equilibrium), then near $\mathbf{x}^*$ the nonlinear system has the same qualitative behavior as its linearization — stable stays stable, saddle stays saddle.

| Linearization says | Nonlinear system near $\mathbf{x}^*$ |
|--------------------|------------------------------------|
| All $\operatorname{Re}\lambda < 0$ | Locally asymptotically stable ✓ |
| Some $\operatorname{Re}\lambda > 0$ | Unstable ✓ |
| Some $\operatorname{Re}\lambda = 0$, none positive | **Inconclusive** — nonlinear terms decide |

Inconclusive example: the frictionless hanging pendulum linearizes to a center ($\pm i\sqrt{g/l}$). Here the true system also oscillates, but for $\dot{x} = -x^3$ versus $\dot{x} = x^3$ the linearization is $\dot{x} = 0$ in both cases while one is stable and the other is not. Use Lyapunov's method (§7.3) instead.

**Practical limits:**

- **"Near" is local.** The pendulum's $\sin\theta \approx \theta$ is within 1% up to about $14^\circ$ and within 5% up to about $31^\circ$.
- **Moving operating points.** For a robot following a trajectory, linearize along the trajectory: $A(t), B(t)$ change at every step. This **time-varying linearization** is the basis of the Extended Kalman Filter, iLQR, and real-time iteration MPC.
- **Gain scheduling.** Precompute linear models at many operating points and switch or interpolate between them.

---

## 7. Stability

### 7.1 Definitions

Let $\mathbf{x}^* = \mathbf{0}$ be an equilibrium of $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$ (shift coordinates if needed).

| Notion | Meaning | Example |
|--------|---------|---------|
| **Lyapunov stable** | Start close enough → stay close forever | Frictionless pendulum (center) |
| **Asymptotically stable** | Lyapunov stable **and** $\mathbf{x}(t) \to \mathbf{0}$ | Damped pendulum |
| **Exponentially stable** | $\|\mathbf{x}(t)\| \leq c\,e^{-\alpha t}\|\mathbf{x}_0\|$ for some $c, \alpha > 0$ | Any stable LTI system |
| **Globally** asymptotically stable | Converges from **every** starting point | $\dot{x} = -x$; not the pendulum (it has another equilibrium) |
| **Unstable** | Not Lyapunov stable | Inverted pendulum |
| **BIBO stable** | Every bounded input gives a bounded output | Stable LTI system; not the double integrator |

For LTI systems asymptotic and exponential stability coincide, and local stability is automatically global.

### 7.2 The Eigenvalue Test

For $\dot{\mathbf{x}} = A\mathbf{x}$, every solution is a combination of $e^{\lambda_i t}$ (times polynomials in $t$ for Jordan blocks), so:

| Eigenvalues of $A$ | Continuous time $\dot{\mathbf{x}} = A\mathbf{x}$ |
|--------------------|---------------------------------------------|
| All $\operatorname{Re}\lambda_i < 0$ | **Asymptotically stable** — $A$ is called **Hurwitz** |
| Any $\operatorname{Re}\lambda_i > 0$ | **Unstable** |
| All $\operatorname{Re}\lambda_i \leq 0$, those on the axis non-repeated (simple) | **Marginally stable** — bounded, no decay |
| A repeated eigenvalue on the axis with a Jordan block | Unstable — grows like $t^k$ (e.g. double integrator: $x = x_0 + v_0 t$) |

**Discrete time** $\mathbf{x}_{k+1} = A_d\mathbf{x}_k$: replace "$\operatorname{Re}\lambda < 0$" with "$|\lambda| < 1$" ([linear_algebra.md](../basics/linear_algebra.md) §14.2). The exact discretization (§8.3) maps $\lambda \mapsto e^{\lambda T}$, which sends the open left half-plane exactly onto the open unit disk.

**How fast:** the slowest mode — the eigenvalue with the **least negative real part** — sets the convergence rate. Its time constant is $1/|\operatorname{Re}\lambda|$.

**Without computing eigenvalues (Routh–Hurwitz, low order):**

| Characteristic polynomial | Stable iff |
|---------------------------|-----------|
| $s^2 + a_1 s + a_0$ | $a_1 > 0$ and $a_0 > 0$ |
| $s^3 + a_2 s^2 + a_1 s + a_0$ | all $a_i > 0$ and $a_2 a_1 > a_0$ |

For the mass–spring–damper ($a_1 = c/m$, $a_0 = k/m$): stable iff damping and stiffness are both positive — physically obvious, now proved.

### 7.3 Lyapunov's Direct Method

Eigenvalues only work for linear systems (or linearizations). Lyapunov's idea works directly on the nonlinear system: find an **energy-like function** that always decreases.

**Theorem.** If there is a function $V(\mathbf{x})$ with

1. $V(\mathbf{0}) = 0$ and $V(\mathbf{x}) > 0$ for $\mathbf{x} \neq \mathbf{0}$ (positive definite), and
2. $\dot{V} = \nabla V^\top\,\mathbf{f}(\mathbf{x}) \leq 0$ along trajectories,

then $\mathbf{0}$ is Lyapunov stable. If $\dot{V} < 0$ for $\mathbf{x} \neq \mathbf{0}$, it is asymptotically stable.

$\dot{V}$ is computed by the chain rule **without solving the ODE** — that is the whole point.

**Example — the damped pendulum** (§6.3 with $u = 0$). Use total mechanical energy:

$$V = \underbrace{\tfrac12 ml^2\dot\theta^2}_{\text{kinetic}} + \underbrace{mgl(1 - \cos\theta)}_{\text{potential}}$$

$$\dot{V} = ml^2\dot\theta\,\ddot\theta + mgl\sin\theta\,\dot\theta = \dot\theta\big(-mgl\sin\theta - b\dot\theta\big) + mgl\sin\theta\,\dot\theta = -b\,\dot\theta^2 \leq 0$$

Friction only removes energy, so the hanging position is stable. ($\dot{V} = 0$ whenever $\dot\theta = 0$, so this $V$ alone gives only "stable"; **LaSalle's invariance principle** completes the argument: the only trajectory that can stay in $\{\dot\theta = 0\}$ near the bottom is the equilibrium itself, so it is asymptotically stable.)

**Linear systems — the Lyapunov equation.** Try a quadratic $V = \mathbf{x}^\top P\mathbf{x}$ with $P \succ 0$. Then $\dot{V} = \mathbf{x}^\top(A^\top P + PA)\mathbf{x}$. Requiring $\dot{V} = -\mathbf{x}^\top Q\mathbf{x}$:

$$\boxed{\ A^\top P + PA = -Q\ }$$

**Theorem:** $A$ is Hurwitz $\iff$ for any $Q \succ 0$ this equation has a unique solution $P \succ 0$ (positive definite: [linear_algebra.md](../basics/linear_algebra.md) §10).

**Worked example.** $A = \begin{bmatrix} 0 & 1 \\ -2 & -3 \end{bmatrix}$, $Q = I$, unknown symmetric $P = \begin{bmatrix} p_1 & p_2 \\ p_2 & p_3 \end{bmatrix}$.

$$A^\top P + PA = \begin{bmatrix} -4p_2 & p_1 - 3p_2 - 2p_3 \\ p_1 - 3p_2 - 2p_3 & 2p_2 - 6p_3 \end{bmatrix} = \begin{bmatrix} -1 & 0 \\ 0 & -1 \end{bmatrix}$$

$p_2 = \tfrac14$, then $p_3 = \tfrac{1 + 2p_2}{6} = \tfrac14$, then $p_1 = 3p_2 + 2p_3 = \tfrac54$:

$$P = \begin{bmatrix} 1.25 & 0.25 \\ 0.25 & 0.25 \end{bmatrix}, \qquad 1.25 > 0,\ \ \det P = 0.3125 - 0.0625 = 0.25 > 0 \ \Rightarrow\ P \succ 0 \ \checkmark$$

So $V = \mathbf{x}^\top P\mathbf{x}$ certifies stability, agreeing with the eigenvalues $-1, -2$ from §4.2.

**Why bother when eigenvalues are easier:** Lyapunov functions extend to nonlinear systems, give **regions** of guaranteed convergence (sublevel sets of $V$), and are the language of robust and adaptive control proofs. The Lyapunov equation also reappears in LQR and in the steady-state covariance of a Kalman filter.

---

## 8. Discretization

### 8.1 Why Discretize

Physics is continuous; computers are not. A controller or estimator runs every $T$ seconds (the **sample time**), reading sensors and updating commands. Between updates, the command is typically **held constant** by the hardware (a **zero-order hold**, ZOH). The goal is an exact or approximate **discrete-time model**:

$$\mathbf{x}_{k+1} = A_d\,\mathbf{x}_k + B_d\,\mathbf{u}_k, \qquad \mathbf{x}_k \equiv \mathbf{x}(kT)$$

| Method | $A_d$ | $B_d$ | Accuracy | Stability preserved? |
|--------|-------|-------|----------|----------------------|
| Forward Euler | $I + AT$ | $BT$ | $O(T)$ | Only if $T$ small enough |
| Backward Euler | $(I - AT)^{-1}$ | $(I - AT)^{-1}BT$ | $O(T)$ | Always (even over-damps) |
| **Exact ZOH** | $e^{AT}$ | $\int_0^T e^{A\sigma}d\sigma\, B$ | Exact for piecewise-constant $\mathbf{u}$ | Exactly |

### 8.2 Forward & Backward Euler

**Forward Euler** replaces the derivative by a forward difference, $\dot{\mathbf{x}}(kT) \approx (\mathbf{x}_{k+1} - \mathbf{x}_k)/T$:

$$\mathbf{x}_{k+1} = \mathbf{x}_k + T\big(A\mathbf{x}_k + B\mathbf{u}_k\big) = (I + AT)\mathbf{x}_k + BT\,\mathbf{u}_k$$

Each continuous eigenvalue $\lambda$ becomes $1 + \lambda T$. Stability needs $|1 + \lambda T| < 1$ — a disk of radius $1/T$ centered at $-1/T$. For a real $\lambda < 0$ this means $T < 2/|\lambda|$.

**Example — $\dot{x} = -10x$** (true factor per step is $e^{-10T}$):

| $T$ | Forward Euler $1 - 10T$ | Exact $e^{-10T}$ | Backward Euler $1/(1 + 10T)$ |
|-----|-------------------------|------------------|------------------------------|
| $0.01$ | $0.900$ | $0.905$ | $0.909$ |
| $0.05$ | $0.500$ | $0.607$ | $0.667$ |
| $0.25$ | $-1.500$ → **diverges, flipping sign** | $0.082$ | $0.286$ |

A perfectly stable system **blows up** in simulation because the step was too large for its fastest mode. This is **numerical instability**, not physics.

**Forward Euler on an oscillator always gains energy.** For $\lambda = \pm i\omega$ (undamped spring), $|1 \pm i\omega T| = \sqrt{1 + \omega^2T^2} > 1$ for every $T > 0$, so the amplitude grows each step. Backward Euler gives $1/\sqrt{1 + \omega^2T^2} < 1$ — it artificially damps.

**Backward Euler** uses the derivative at the **new** point, $(\mathbf{x}_{k+1} - \mathbf{x}_k)/T = A\mathbf{x}_{k+1} + B\mathbf{u}_k$, so a linear solve is needed each step:

$$\mathbf{x}_{k+1} = (I - AT)^{-1}\big(\mathbf{x}_k + BT\,\mathbf{u}_k\big)$$

Eigenvalues map to $1/(1 - \lambda T)$, which lies inside the unit circle for **every** $\operatorname{Re}\lambda < 0$ and every $T > 0$. This is why implicit methods are used for **stiff** systems (fast and slow modes mixed — e.g., stiff contact springs alongside slow arm motion).

**Semi-implicit (symplectic) Euler** — the default in many physics engines. Update velocity first, then use the **new** velocity for position:

$$v_{k+1} = v_k + T\,a(x_k, v_k), \qquad x_{k+1} = x_k + T\,v_{k+1}$$

Same cost as forward Euler, but for oscillators it keeps energy bounded instead of growing it.

### 8.3 Exact Discretization (Zero-Order Hold)

If $\mathbf{u}(t) = \mathbf{u}_k$ is constant on $[kT, (k+1)T)$, apply the solution formula of §5.2 over one interval:

$$\mathbf{x}_{k+1} = e^{AT}\mathbf{x}_k + \int_{kT}^{(k+1)T} e^{A((k+1)T - s)}B\, ds\;\mathbf{u}_k$$

Substituting $\sigma = (k+1)T - s$:

$$\boxed{\ A_d = e^{AT}, \qquad B_d = \left(\int_0^T e^{A\sigma}\, d\sigma\right) B \ \ \Big(= A^{-1}\big(e^{AT} - I\big)B \text{ if } A \text{ is invertible}\Big)\ }$$

This is **exact** at the sample instants — no approximation beyond the ZOH assumption itself. Eigenvalues map as $\lambda_d = e^{\lambda T}$, so stability is preserved exactly.

**Example 1 — first-order system** $\dot{x} = -a x + b u$:

$$A_d = e^{-aT}, \qquad B_d = \frac{b}{a}\big(1 - e^{-aT}\big)$$

Steady state check: $x = A_d x + B_d u \Rightarrow x = \tfrac{B_d}{1 - A_d}u = \tfrac{b}{a}u$ — same as continuous time. ✓

**Example 2 — double integrator** $\ddot{p} = u$, state $(p, v)$. From §4.2, $e^{A\sigma} = \begin{bmatrix} 1 & \sigma \\ 0 & 1 \end{bmatrix}$:

$$A_d = \begin{bmatrix} 1 & T \\ 0 & 1 \end{bmatrix}, \qquad B_d = \int_0^T \begin{bmatrix} 1 & \sigma \\ 0 & 1 \end{bmatrix}\begin{bmatrix} 0 \\ 1 \end{bmatrix} d\sigma = \int_0^T \begin{bmatrix} \sigma \\ 1 \end{bmatrix} d\sigma = \begin{bmatrix} T^2/2 \\ T \end{bmatrix}$$

$$p_{k+1} = p_k + T v_k + \tfrac{T^2}{2}u_k, \qquad v_{k+1} = v_k + T u_k$$

— the familiar constant-acceleration kinematics. Forward Euler gives the same $A_d$ but $B_d = (0, T)$: it **misses the $\tfrac{T^2}{2}u$ term**, so position lags by half a step of acceleration.

### 8.4 Computing A_d and B_d in Practice

**Van Loan's block-matrix trick** computes both at once, and works even when $A$ is singular (as in the double integrator or DC motor):

$$\exp\!\left(\begin{bmatrix} A & B \\ 0 & 0 \end{bmatrix} T\right) = \begin{bmatrix} A_d & B_d \\ 0 & I \end{bmatrix}$$

Check with the double integrator: $M = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{bmatrix}$, $M^3 = 0$, so $e^{MT} = I + MT + \tfrac{(MT)^2}{2} = \begin{bmatrix} 1 & T & T^2/2 \\ 0 & 1 & T \\ 0 & 0 & 1 \end{bmatrix}$ — top-left block $A_d$, top-right $B_d$. ✓

```python
import numpy as np
from scipy.linalg import expm

def c2d_zoh(A, B, T):
    n, p = B.shape
    M = np.zeros((n + p, n + p))
    M[:n, :n], M[:n, n:] = A, B
    E = expm(M * T)
    return E[:n, :n], E[:n, n:]          # A_d, B_d

A = np.array([[0., 1.], [0., 0.]]); B = np.array([[0.], [1.]])
Ad, Bd = c2d_zoh(A, B, 0.1)              # Ad = [[1, .1], [0, 1]], Bd = [[.005], [.1]]
```

`scipy.signal.cont2discrete((A, B, C, D), T, method='zoh')` does the same for a full model.

> **Never compute $e^{AT}$ by summing the raw Taylor series for large $\|AT\|$** — it is numerically unreliable. Library `expm` uses scaling-and-squaring with Padé approximants.

### 8.5 Choosing the Sample Time

| Guideline | Reason |
|-----------|--------|
| $T \ll 1/|\lambda_{\max}|$ — several samples per fastest time constant | Capture the fastest dynamics you care about |
| Sample rate ≈ 10–30× the desired closed-loop bandwidth | Common control rule of thumb |
| Sample faster than unstable modes diverge | Inverted pendulum (§6.3) doubles every 0.22 s → control at ≥ 100 Hz is comfortable |
| ZOH adds roughly $T/2$ of effective delay | Delay erodes stability margins, so slower sampling hurts even with exact discretization |
| Above twice the highest frequency in the signal (Nyquist) | Avoid aliasing — the absolute floor, not a target |

**Typical rates in robotics:** joint torque/current loops 1–10 kHz; joint position loops and whole-body control 0.5–1 kHz; state estimation 100 Hz–1 kHz; MPC 20–500 Hz; high-level planning 1–10 Hz.

### 8.6 Numerical Integrators for Simulation

For nonlinear $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u})$ there is no closed-form $e^{AT}$ — simulate with a numerical integrator. The workhorse beyond Euler is **4th-order Runge–Kutta (RK4)**:

$$\begin{aligned}
\mathbf{k}_1 &= \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k) \\
\mathbf{k}_2 &= \mathbf{f}(\mathbf{x}_k + \tfrac{T}{2}\mathbf{k}_1, \mathbf{u}_k) \\
\mathbf{k}_3 &= \mathbf{f}(\mathbf{x}_k + \tfrac{T}{2}\mathbf{k}_2, \mathbf{u}_k) \\
\mathbf{k}_4 &= \mathbf{f}(\mathbf{x}_k + T\mathbf{k}_3, \mathbf{u}_k) \\
\mathbf{x}_{k+1} &= \mathbf{x}_k + \tfrac{T}{6}\big(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4\big)
\end{aligned}$$

| Integrator | Global error | $\mathbf{f}$ evaluations per step | Typical use |
|------------|-------------|------------------------------|-------------|
| Forward Euler | $O(T)$ | 1 | Quick prototypes; only with very small $T$ |
| Semi-implicit Euler | $O(T)$ | 1 | Game/robot physics engines (e.g. MuJoCo, PyBullet) at ~1 kHz |
| RK4 | $O(T^4)$ | 4 | Accurate offline simulation, MPC prediction models |
| Adaptive (RK45 / `solve_ivp`) | User tolerance | Varies | Offline analysis where accuracy matters more than timing |
| Implicit (backward Euler, BDF) | $O(T)$ – $O(T^5)$ | Solve per step | Stiff systems: contact, soft bodies, stiff actuators |

Halving $T$ cuts Euler's error in half but RK4's by $2^4 = 16$.

```python
def rk4_step(f, x, u, T):
    k1 = f(x, u)
    k2 = f(x + T/2 * k1, u)
    k3 = f(x + T/2 * k2, u)
    k4 = f(x + T * k3, u)
    return x + T/6 * (k1 + 2*k2 + 2*k3 + k4)
```

---

## 9. Robotics Applications

### 9.1 Manipulator Dynamics in State-Space Form

The equations of motion of an $n$-joint rigid robot:

$$M(\mathbf{q})\,\ddot{\mathbf{q}} + C(\mathbf{q}, \dot{\mathbf{q}})\,\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol\tau$$

| Term | Shape | Meaning |
|------|-------|---------|
| $M(\mathbf{q})$ | $n \times n$ | Mass (inertia) matrix — symmetric positive definite |
| $C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}}$ | $n \times 1$ | Coriolis and centrifugal forces — quadratic in $\dot{\mathbf{q}}$ |
| $\mathbf{g}(\mathbf{q})$ | $n \times 1$ | Gravity torques |
| $\boldsymbol\tau$ | $n \times 1$ | Joint torques (the input) |

**State-space form** with $\mathbf{x} = (\mathbf{q}, \dot{\mathbf{q}}) \in \mathbb{R}^{2n}$ and $\mathbf{u} = \boldsymbol\tau$ (§4.1):

$$\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u}) = \begin{bmatrix} \dot{\mathbf{q}} \\ M(\mathbf{q})^{-1}\big(\boldsymbol\tau - C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} - \mathbf{g}(\mathbf{q})\big) \end{bmatrix}$$

($M$ is positive definite, so it is always invertible.)

**Linearization at a static pose** $\mathbf{q}^*$ with $\dot{\mathbf{q}}^* = \mathbf{0}$ and holding torque $\boldsymbol\tau^* = \mathbf{g}(\mathbf{q}^*)$ (§6.1):

- $C\dot{\mathbf{q}}$ is quadratic in $\dot{\mathbf{q}}$, so its derivatives vanish at $\dot{\mathbf{q}} = \mathbf{0}$.
- $\partial M^{-1}/\partial\mathbf{q}$ multiplies $(\boldsymbol\tau^* - \mathbf{g}(\mathbf{q}^*)) = \mathbf{0}$, so it vanishes too.

$$A = \begin{bmatrix} 0 & I \\ -M(\mathbf{q}^*)^{-1}\dfrac{\partial\mathbf{g}}{\partial\mathbf{q}}\Big|_{\mathbf{q}^*} & 0 \end{bmatrix}, \qquad B = \begin{bmatrix} 0 \\ M(\mathbf{q}^*)^{-1} \end{bmatrix}$$

For a single pendulum ($M = ml^2$, $g(q) = mgl\sin q$) this reproduces §6.3 exactly. The gravity Jacobian $\partial\mathbf{g}/\partial\mathbf{q}$ plays the role of a (possibly negative) **stiffness matrix**.

### 9.2 The Simulation Loop

Every robot simulator is, at its core, a discretized $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u})$:

```python
x = x0
for k in range(N):
    y   = sensor_model(x) + noise()          # what the robot measures
    u   = controller(y, k * T)               # runs at the control rate
    for _ in range(substeps):                # physics often runs faster than control
        x = integrator_step(f, x, u, T / substeps)   # Euler / semi-implicit / RK4 (§8)
```

Design choices map directly to this document: the **integrator and step size** (§8.2, §8.6) set accuracy and numerical stability; **substeps** let the physics resolve stiff contact while the controller runs at its real rate; holding $\mathbf{u}$ constant inside the inner loop is the **ZOH** of §8.3. The sim-to-real gap partly comes from these choices — a too-large step can make a simulated system look more (or less) stable than the real one.

### 9.3 Kalman Filter — the Prediction Step

A Kalman filter tracks a Gaussian belief $\mathcal{N}(\hat{\mathbf{x}}, P)$ over the state. Between measurements it **predicts** using the discretized model:

$$\hat{\mathbf{x}}_{k+1|k} = A_d\,\hat{\mathbf{x}}_{k|k} + B_d\,\mathbf{u}_k, \qquad P_{k+1|k} = A_d\,P_{k|k}\,A_d^\top + Q_d$$

- The mean propagates through the dynamics (§8.3).
- The covariance propagates by the affine-transformation rule $\Sigma \to A\Sigma A^\top$ ([statistics_probability.md](../basics/statistics_probability.md) §4.3), plus **process noise** $Q_d$ for model error and disturbances.
- For continuous white noise with intensity $Q_c$ entering through $G$: $Q_d = \int_0^T e^{A\sigma}GQ_cG^\top e^{A^\top\sigma}d\sigma \approx GQ_cG^\top T$ for small $T$ (also computable with a Van Loan block exponential).
- **Extended Kalman Filter:** for nonlinear $\mathbf{f}$, propagate the mean with the nonlinear model (e.g. RK4) and the covariance with the **Jacobian** $A_d \approx \partial\mathbf{f}_d/\partial\mathbf{x}$ evaluated at the current estimate (§6.4 time-varying linearization).

If $A_d$ is unstable (e.g. the inverted pendulum), $P$ grows between measurements — uncertainty literally follows the dynamics.

### 9.4 MPC — the Prediction Model

Model Predictive Control chooses the next $N$ inputs by predicting the future with $\mathbf{x}_{k+1} = A_d\mathbf{x}_k + B_d\mathbf{u}_k$. Unrolling:

$$\mathbf{x}_{k+i} = A_d^{\,i}\,\mathbf{x}_k + \sum_{j=0}^{i-1} A_d^{\,i-1-j}B_d\,\mathbf{u}_{k+j}$$

Stacking all $N$ steps gives one linear map from the current state and the input sequence to the predicted trajectory:

$$\underbrace{\begin{bmatrix} \mathbf{x}_{k+1} \\ \mathbf{x}_{k+2} \\ \vdots \\ \mathbf{x}_{k+N} \end{bmatrix}}_{\mathbf{X}} = \underbrace{\begin{bmatrix} A_d \\ A_d^2 \\ \vdots \\ A_d^N \end{bmatrix}}_{\mathcal{A}}\mathbf{x}_k + \underbrace{\begin{bmatrix} B_d & & & \\ A_dB_d & B_d & & \\ \vdots & & \ddots & \\ A_d^{N-1}B_d & \cdots & A_dB_d & B_d \end{bmatrix}}_{\mathcal{B}}\underbrace{\begin{bmatrix} \mathbf{u}_k \\ \mathbf{u}_{k+1} \\ \vdots \\ \mathbf{u}_{k+N-1} \end{bmatrix}}_{\mathbf{U}}$$

With a quadratic cost on $\mathbf{X}$ and $\mathbf{U}$ and linear limits on torques and states, choosing $\mathbf{U}$ is a **quadratic program** ([optimization.md](../basics/optimization.md) §10.2, §12.4). Only $\mathbf{u}_k$ is applied; at the next sample the state is re-measured and the problem re-solved (**receding horizon**). For nonlinear robots the same structure is rebuilt each cycle from a linearization along the previous solution (§6.4).

---

## Summary — What to Master First

| Priority | Topic | Because |
|----------|-------|---------|
| 1 | Writing any model as $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u})$; state-space form (§4.1, §5.1) | Every simulator, estimator and controller starts here |
| 2 | Eigenvalues ↔ behavior: decay, oscillation, stability (§2.2, §4.2, §7.2) | Read a model's behavior without simulating it |
| 3 | Equilibria and linearization with Jacobians (§6) | Turns nonlinear robots into tractable linear models |
| 4 | Exact ZOH discretization and Euler's pitfalls (§8.2–8.4) | Bridges the continuous model and the code that runs at $T$ seconds |
| 5 | Laplace transform: derivative rule, partial fractions, transfer functions (§3) | Fastest hand method for forced ODEs; the language of poles and control design |
| 5 | Damping ratio and natural frequency (§2.3) | Common language for tuning and specifying joint response |
| 6 | Lyapunov functions (§7.3) | Stability proofs for nonlinear systems; foundation for LQR and robust control |

**Companion documents:** [linear_algebra.md](../basics/linear_algebra.md) (eigenvalues, matrix exponential, similarity, discrete linear systems) · [multivariable_calculus.md](../basics/multivariable_calculus.md) (Jacobians, Taylor expansion, matrix calculus) · [statistics_probability.md](../basics/statistics_probability.md) (Gaussians for the Kalman filter) · [optimization.md](../basics/optimization.md) (QP and trajectory optimization for MPC) · [signals_systems.md](signals_systems.md) (Fourier, frequency response, sampling, z-transform, filters)
