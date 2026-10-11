# Signals & Systems

A robot is surrounded by signals: encoder angles, IMU accelerations, motor currents, camera timestamps. Every one of them is noisy, sampled, and passed through some system — a sensor, a filter, a controller, the robot's own dynamics. This document gives the **frequency-domain** view of those signals and systems: decompose a signal into sinusoids (Fourier), predict how a system scales and delays each sinusoid (frequency response), understand what sampling does (Nyquist, aliasing), model discrete-time systems ($z$-transform), analyze real data (FFT), and design the filters that clean it up.

**Prerequisites:** complex numbers and Euler's formula $e^{i\theta} = \cos\theta + i\sin\theta$; the Laplace transform, transfer functions, and discretization from [ode_linear_systems.md](ode_linear_systems.md) §3, §5, §8.

**Convention:** angular frequency $\omega$ in rad/s; ordinary frequency $f = \omega / 2\pi$ in Hz. Sample time $T$, sample rate $f_s = 1/T$.

## Table of Contents

1. [Signals & LTI Systems](#1-signals--lti-systems)
   - 1.1 [Basic Signals](#11-basic-signals)
   - 1.2 [Linear Time-Invariant Systems](#12-linear-time-invariant-systems)
   - 1.3 [Impulse Response & Convolution](#13-impulse-response--convolution)
   - 1.4 [Complex Exponentials Are Eigenfunctions](#14-complex-exponentials-are-eigenfunctions)
2. [Fourier Series](#2-fourier-series)
   - 2.1 [Periodic Signals as Sums of Sinusoids](#21-periodic-signals-as-sums-of-sinusoids)
   - 2.2 [Computing the Coefficients](#22-computing-the-coefficients)
   - 2.3 [Convergence, Gibbs & Parseval](#23-convergence-gibbs--parseval)
3. [The Fourier Transform](#3-the-fourier-transform)
   - 3.1 [From Series to Transform](#31-from-series-to-transform)
   - 3.2 [Transform Pairs](#32-transform-pairs)
   - 3.3 [Properties](#33-properties)
   - 3.4 [Fourier vs Laplace](#34-fourier-vs-laplace)
   - 3.5 [Time–Frequency Uncertainty](#35-timefrequency-uncertainty)
4. [Frequency Response](#4-frequency-response)
   - 4.1 [Sinusoid In, Sinusoid Out](#41-sinusoid-in-sinusoid-out)
   - 4.2 [Bode Plots](#42-bode-plots)
   - 4.3 [First- and Second-Order Building Blocks](#43-first--and-second-order-building-blocks)
   - 4.4 [Bandwidth & Speed](#44-bandwidth--speed)
5. [Sampling](#5-sampling)
   - 5.1 [What Sampling Does to the Spectrum](#51-what-sampling-does-to-the-spectrum)
   - 5.2 [The Nyquist–Shannon Theorem](#52-the-nyquistshannon-theorem)
   - 5.3 [Aliasing](#53-aliasing)
   - 5.4 [Anti-Aliasing & Reconstruction](#54-anti-aliasing--reconstruction)
6. [The z-Transform](#6-the-z-transform)
   - 6.1 [Definition & Pairs](#61-definition--pairs)
   - 6.2 [Properties & Difference Equations](#62-properties--difference-equations)
   - 6.3 [Discrete Transfer Functions & Poles](#63-discrete-transfer-functions--poles)
   - 6.4 [From s to z](#64-from-s-to-z)
7. [The DFT & FFT](#7-the-dft--fft)
   - 7.1 [The Discrete Fourier Transform](#71-the-discrete-fourier-transform)
   - 7.2 [Frequency Bins & Resolution](#72-frequency-bins--resolution)
   - 7.3 [The FFT Algorithm](#73-the-fft-algorithm)
   - 7.4 [Practical Spectrum Analysis](#74-practical-spectrum-analysis)
8. [Filters](#8-filters)
   - 8.1 [Filter Types & Specifications](#81-filter-types--specifications)
   - 8.2 [First-Order Low-Pass, Continuous & Discrete](#82-first-order-low-pass-continuous--discrete)
   - 8.3 [Butterworth & Higher-Order Filters](#83-butterworth--higher-order-filters)
   - 8.4 [FIR vs IIR](#84-fir-vs-iir)
   - 8.5 [Delay: the Price of Filtering](#85-delay-the-price-of-filtering)
9. [Robotics Applications](#9-robotics-applications)
   - 9.1 [Complementary Filter for IMU Attitude](#91-complementary-filter-for-imu-attitude)
   - 9.2 [Differentiating Noisy Encoder Signals](#92-differentiating-noisy-encoder-signals)
   - 9.3 [Vibration Analysis & Notch Filters](#93-vibration-analysis--notch-filters)
   - 9.4 [Measuring a Frequency Response](#94-measuring-a-frequency-response)

---

## 1. Signals & LTI Systems

### 1.1 Basic Signals

A **continuous-time** signal $x(t)$ is defined for every real $t$; a **discrete-time** signal $x[k]$ is a sequence, usually samples $x[k] = x(kT)$.

| Signal | Continuous | Discrete | Role |
|--------|-----------|----------|------|
| Unit impulse | $\delta(t)$: zero except at $0$, $\int \delta\,dt = 1$ | $\delta[k] = 1$ if $k = 0$, else $0$ | Probe that reveals a system completely (§1.3) |
| Unit step | $u(t) = 1$ for $t \geq 0$ | $u[k] = 1$ for $k \geq 0$ | Switch-on; setpoint change |
| Exponential | $e^{st}$, $s \in \mathbb{C}$ | $z^k$, $z \in \mathbb{C}$ | Eigenfunctions of LTI systems (§1.4) |
| Sinusoid | $A\cos(\omega t + \phi)$ | $A\cos(\Omega k + \phi)$ | Building block of Fourier analysis |

The impulse is the derivative of the step, and its **sifting property** $\int x(\tau)\,\delta(t - \tau)\,d\tau = x(t)$ says any signal is a continuous sum of shifted, scaled impulses.

A discrete sinusoid's frequency $\Omega = \omega T$ is in **radians per sample**. Since $\cos((\Omega + 2\pi)k) = \cos(\Omega k)$ for integer $k$, discrete frequencies are only distinct within $[-\pi, \pi]$ — the root of aliasing (§5.3).

### 1.2 Linear Time-Invariant Systems

A **system** maps an input signal $x$ to an output signal $y$.

| Property | Definition | Fails for |
|----------|-----------|-----------|
| **Linear** | Input $a x_1 + b x_2$ → output $a y_1 + b y_2$ | Saturation, friction, $y = x^2$ |
| **Time-invariant** | Input delayed by $t_0$ → output delayed by $t_0$ | A system whose parameters change with time |
| **Causal** | Output at $t$ depends only on inputs up to $t$ | Offline smoothing filters (§8.5) |
| **BIBO stable** | Bounded input → bounded output | Integrator, any system with a pole in the right half-plane |

Every linear constant-coefficient ODE with zero initial conditions is an LTI system ([ode_linear_systems.md](ode_linear_systems.md) §3.5). Real robots are only approximately LTI — near an operating point (linearization) and below saturation.

### 1.3 Impulse Response & Convolution

The **impulse response** $h(t)$ is the output when the input is $\delta(t)$. For an LTI system it determines the output for **every** input:

$$y(t) = (h * x)(t) = \int_{-\infty}^{\infty} h(\tau)\, x(t - \tau)\, d\tau \qquad\qquad y[k] = \sum_{j} h[j]\, x[k - j]$$

**Why:** the input is a sum of shifted, scaled impulses (§1.1); linearity and time-invariance make the output the same sum of shifted, scaled impulse responses.

**Reading the formula:** the output now is a weighted sum of past inputs, with weight $h(\tau)$ on the input from $\tau$ seconds ago. $h$ is the system's **memory profile**.

**Discrete example.** $h = [1,\ 0.5]$ (a sample plus half of the previous one), $x = [1,\ 2,\ 3]$:

| $k$ | $y[k] = h[0]x[k] + h[1]x[k-1]$ |
|-----|-------------------------------|
| 0 | $1 \cdot 1 = 1$ |
| 1 | $1 \cdot 2 + 0.5 \cdot 1 = 2.5$ |
| 2 | $1 \cdot 3 + 0.5 \cdot 2 = 4$ |
| 3 | $0.5 \cdot 3 = 1.5$ |

Output length = $3 + 2 - 1 = 4$. In numpy: `np.convolve([1, 0.5], [1, 2, 3])`.

### 1.4 Complex Exponentials Are Eigenfunctions

Feed $x(t) = e^{st}$ into an LTI system with impulse response $h$:

$$y(t) = \int h(\tau)\,e^{s(t - \tau)}\,d\tau = e^{st}\underbrace{\int h(\tau)\,e^{-s\tau}\,d\tau}_{H(s)} = H(s)\,e^{st}$$

The output is the **same exponential, scaled by a complex number** $H(s)$ — exactly like an eigenvector $A\mathbf{v} = \lambda\mathbf{v}$, with $e^{st}$ as the eigenfunction and $H(s)$ (the transfer function, i.e. the Laplace transform of $h$) as the eigenvalue. In discrete time, $z^k \mapsto H(z)\,z^k$.

> **Key insight:** this is why the whole subject works. Decompose any input into complex exponentials (Fourier, §2–3), multiply each by $H$, and add them back up. Convolution in time becomes **multiplication** in frequency. Every transform in this document is a change of basis into the eigenfunctions of LTI systems.

---

## 2. Fourier Series

### 2.1 Periodic Signals as Sums of Sinusoids

A signal with period $T_0$ (fundamental frequency $\omega_0 = 2\pi/T_0$) can be written as a sum of harmonics at integer multiples of $\omega_0$:

$$x(t) = a_0 + \sum_{n=1}^{\infty}\big(a_n\cos n\omega_0 t + b_n\sin n\omega_0 t\big) = \sum_{n=-\infty}^{\infty} c_n\, e^{in\omega_0 t}$$

| Form | Coefficients | Notes |
|------|-------------|-------|
| Trigonometric | $a_0,\ a_n,\ b_n$ (real) | Intuitive |
| Complex exponential | $c_n$ (complex), $n \in \mathbb{Z}$ | Compact; $c_0 = a_0$, $c_n = \tfrac12(a_n - i\,b_n)$, $c_{-n} = \overline{c_n}$ for real $x$ |

$|c_n|$ is the **amplitude** and $\angle c_n$ the **phase** of harmonic $n$. Plotting $|c_n|$ against $n\omega_0$ gives the (line) **spectrum**.

### 2.2 Computing the Coefficients

The exponentials $e^{in\omega_0 t}$ are **orthogonal** over one period — $\int_{T_0} e^{in\omega_0 t}e^{-im\omega_0 t}\,dt = T_0$ if $n = m$, else $0$ — so each coefficient is a projection, just like coordinates in an orthonormal basis ([linear_algebra.md](../basics/linear_algebra.md) §9.2):

$$c_n = \frac{1}{T_0}\int_{T_0} x(t)\, e^{-in\omega_0 t}\, dt, \qquad a_n = \frac{2}{T_0}\int_{T_0} x(t)\cos n\omega_0 t\, dt, \qquad b_n = \frac{2}{T_0}\int_{T_0} x(t)\sin n\omega_0 t\, dt$$

**Worked example — square wave.** Period $2\pi$ ($\omega_0 = 1$), $x = +1$ on $(0, \pi)$ and $-1$ on $(-\pi, 0)$. The signal is odd, so $a_n = 0$ and

$$b_n = \frac{2}{2\pi}\int_{-\pi}^{\pi} x(t)\sin nt\, dt = \frac{2}{\pi}\int_0^{\pi}\sin nt\, dt = \frac{2}{\pi}\cdot\frac{1 - \cos n\pi}{n} = \begin{cases} \dfrac{4}{n\pi} & n \text{ odd} \\ 0 & n \text{ even} \end{cases}$$

$$x(t) = \frac{4}{\pi}\left(\sin t + \frac{\sin 3t}{3} + \frac{\sin 5t}{5} + \cdots\right)$$

**Smoothness ↔ decay rate.** The square wave's coefficients fall like $1/n$ because of its jumps. A continuous signal with corners (triangle wave) falls like $1/n^2$; a smooth signal falls faster than any power. Sharp features in time need many high frequencies.

### 2.3 Convergence, Gibbs & Parseval

**Gibbs phenomenon.** At a jump, truncated Fourier sums overshoot by about **9% of the jump size**, and adding more terms makes the overshoot narrower but not smaller. The same ringing appears when a sharp-edged signal passes through a filter that cuts high frequencies abruptly.

**Parseval's theorem** — power is the same in both domains:

$$\frac{1}{T_0}\int_{T_0} |x(t)|^2\, dt = \sum_{n=-\infty}^{\infty} |c_n|^2$$

**Check with the square wave:** the left side is $1$. On the right, each odd $n$ contributes $|c_n|^2 + |c_{-n}|^2 = 2\big(\tfrac{2}{n\pi}\big)^2$, so the sum is $\tfrac{8}{\pi^2}\sum_{n \text{ odd}} \tfrac{1}{n^2} = \tfrac{8}{\pi^2}\cdot\tfrac{\pi^2}{8} = 1$ ✓. The fundamental alone carries $8/\pi^2 \approx 81\%$ of the power.

---

## 3. The Fourier Transform

### 3.1 From Series to Transform

Let the period $T_0 \to \infty$: harmonics $n\omega_0$ crowd together into a continuum, and the sum becomes an integral. For a non-periodic signal:

$$X(\omega) = \mathcal{F}\{x(t)\} = \int_{-\infty}^{\infty} x(t)\, e^{-i\omega t}\, dt \qquad\qquad x(t) = \frac{1}{2\pi}\int_{-\infty}^{\infty} X(\omega)\, e^{i\omega t}\, d\omega$$

$X(\omega)$ is the **spectrum**: $|X(\omega)|$ says how much of frequency $\omega$ the signal contains, $\angle X(\omega)$ its phase.

> **Convention note:** in Hz, $X(f) = \int x(t)\,e^{-i2\pi f t}\,dt$ and the inverse has no $\tfrac{1}{2\pi}$. Both conventions are common — check which one a formula sheet uses before mixing them.

### 3.2 Transform Pairs

| $x(t)$ | $X(\omega)$ | Notes |
|--------|------------|-------|
| $\delta(t)$ | $1$ | An impulse contains **all** frequencies equally |
| $1$ | $2\pi\,\delta(\omega)$ | A constant is pure DC |
| $e^{i\omega_0 t}$ | $2\pi\,\delta(\omega - \omega_0)$ | One frequency |
| $\cos\omega_0 t$ | $\pi\big[\delta(\omega - \omega_0) + \delta(\omega + \omega_0)\big]$ | Two spikes at $\pm\omega_0$ |
| $e^{-at}u(t)$, $a > 0$ | $\dfrac{1}{a + i\omega}$ | First-order system's impulse response |
| Rectangle: $1$ for $\lvert t \rvert < \tfrac{\tau}{2}$ | $\tau\,\dfrac{\sin(\omega\tau/2)}{\omega\tau/2}$ | Sinc; first zero at $\omega = 2\pi/\tau$ |
| Gaussian $e^{-t^2/2\sigma^2}$ | $\sigma\sqrt{2\pi}\,e^{-\sigma^2\omega^2/2}$ | Gaussian ↔ Gaussian, widths inverse |

### 3.3 Properties

| Property | Time domain | Frequency domain |
|----------|------------|-----------------|
| Linearity | $a x + b y$ | $aX + bY$ |
| Time shift | $x(t - t_0)$ | $e^{-i\omega t_0}X(\omega)$ — delay = linear phase |
| Frequency shift (modulation) | $e^{i\omega_0 t}x(t)$ | $X(\omega - \omega_0)$ |
| Time scaling | $x(at)$ | $\tfrac{1}{\lvert a \rvert}X(\omega/a)$ — compress in time = stretch in frequency |
| Differentiation | $\dot{x}(t)$ | $i\omega\,X(\omega)$ — boosts high frequencies |
| **Convolution** | $(h * x)(t)$ | $H(\omega)\,X(\omega)$ |
| Multiplication | $x(t)\,w(t)$ | $\tfrac{1}{2\pi}(X * W)(\omega)$ — windowing smears the spectrum (§7.4) |
| Parseval | $\int \lvert x \rvert^2 dt$ | $\tfrac{1}{2\pi}\int \lvert X \rvert^2 d\omega$ |

The two most-used lines: **convolution ↔ multiplication** (filtering is multiplication of spectra) and **differentiation ↔ multiplication by $i\omega$** (differentiating amplifies noise, §9.2).

### 3.4 Fourier vs Laplace

The Laplace transform ([ode_linear_systems.md](ode_linear_systems.md) §3) uses $e^{-st}$ with $s = \sigma + i\omega$; the Fourier transform uses only $e^{-i\omega t}$. For a causal signal whose Laplace region of convergence includes the imaginary axis (a **stable** impulse response):

$$X(\omega) = F(s)\big|_{s = i\omega}$$

Example: $e^{-at}u(t)$ has $F(s) = \tfrac{1}{s + a}$ and $X(\omega) = \tfrac{1}{i\omega + a}$ ✓.

| | Laplace | Fourier |
|---|---|---|
| Variable | $s = \sigma + i\omega$ (complex plane) | $i\omega$ (imaginary axis only) |
| Handles growing signals / unstable systems | Yes ($\sigma$ large enough) | No (integral diverges) |
| Initial conditions, transients | Built in | Not directly |
| Best for | Solving ODEs, poles, stability | Spectra, steady-state sinusoidal response, filtering, data analysis |

### 3.5 Time–Frequency Uncertainty

A signal cannot be narrow in both time and frequency. With $\Delta t$ and $\Delta\omega$ the standard deviations of $|x|^2$ and $|X|^2$:

$$\Delta t\,\Delta\omega \geq \tfrac12$$

with equality only for the Gaussian. Consequences:

- A **short impulse** (hammer tap) excites a **broad** band — the basis of impact testing for structural resonances.
- To **resolve** two close frequencies you must record **long enough**: resolution $\approx 1/(\text{record length})$ (§7.2).
- Fast-reacting filters cannot be very selective, and sharp filters must have long memory (and delay, §8.5).

---

## 4. Frequency Response

### 4.1 Sinusoid In, Sinusoid Out

For a stable LTI system with transfer function $H(s)$, a sinusoidal input produces — after transients die out — a sinusoid of the **same frequency**, scaled and shifted:

$$x(t) = \sin\omega t \quad\Longrightarrow\quad y_{ss}(t) = |H(i\omega)|\,\sin\big(\omega t + \angle H(i\omega)\big)$$

This is §1.4 with $s = i\omega$ (and taking the imaginary part, since $h$ is real). The complex number $H(i\omega)$ is the **frequency response**: its magnitude is the **gain** and its angle the **phase shift** at that frequency.

**Example — first-order low-pass** $H(s) = \dfrac{1}{\tau s + 1}$, $\tau = 0.1\ \text{s}$:

$$|H(i\omega)| = \frac{1}{\sqrt{1 + (\omega\tau)^2}}, \qquad \angle H(i\omega) = -\arctan(\omega\tau)$$

| Input frequency | $\omega\tau$ | Gain | Phase | Output |
|-----------------|-------------|------|-------|--------|
| 1 Hz ($\omega = 6.28$) | 0.628 | 0.847 | $-32.1^\circ$ | Slightly reduced, slightly late |
| 1.59 Hz ($\omega = 1/\tau = 10$) | 1 | 0.707 ($-3$ dB) | $-45^\circ$ | Corner frequency |
| 10 Hz ($\omega = 62.8$) | 6.28 | 0.157 | $-81.0^\circ$ | Strongly attenuated |

### 4.2 Bode Plots

A **Bode plot** draws gain and phase against frequency on a log axis:

- **Magnitude in decibels:** $|H|_{\text{dB}} = 20\log_{10}|H|$. Gain $10 \to 20$ dB, $2 \to 6$ dB, $1 \to 0$ dB, $0.707 \to -3$ dB, $0.1 \to -20$ dB.
- **Frequency on a log scale**, measured in **decades** (factors of 10).

**Why logs:** a cascade multiplies transfer functions, $H = H_1H_2H_3$. On a Bode plot the dB magnitudes and the phases simply **add**, so a complicated system is sketched by adding a few standard pieces, each approximated by straight-line **asymptotes**.

### 4.3 First- and Second-Order Building Blocks

| Block | Magnitude | Phase |
|-------|-----------|-------|
| Gain $K$ | Flat at $20\log_{10}K$ | $0^\circ$ (or $-180^\circ$ if $K < 0$) |
| Integrator $\tfrac1s$ | $-20$ dB/decade, $0$ dB at $\omega = 1$ | $-90^\circ$ |
| Differentiator $s$ | $+20$ dB/decade | $+90^\circ$ |
| First-order pole $\tfrac{1}{\tau s + 1}$ | Flat, then $-20$ dB/decade after $\omega_c = 1/\tau$; $-3$ dB at $\omega_c$ | $0^\circ \to -90^\circ$; $-45^\circ$ at $\omega_c$ |
| Second-order $\tfrac{\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2}$ | Flat, then $-40$ dB/decade after $\omega_n$; resonant peak if $\zeta < 0.707$ | $0^\circ \to -180^\circ$; $-90^\circ$ at $\omega_n$ |
| Time delay $e^{-sT_d}$ | $0$ dB (no change) | $-\omega T_d$ — unbounded phase loss |

**Second-order resonance.** For $\zeta < 1/\sqrt2$ the gain peaks at $\omega_r = \omega_n\sqrt{1 - 2\zeta^2}$ with height

$$M_r = \frac{1}{2\zeta\sqrt{1 - \zeta^2}}$$

| $\zeta$ | 0.1 | 0.3 | 0.5 | 0.707 |
|---------|-----|-----|-----|-------|
| $M_r$ | 5.03 (14.0 dB) | 1.75 (4.9 dB) | 1.15 (1.2 dB) | 1.00 (0 dB) |

**Worked example — the mass–spring–damper** of [ode_linear_systems.md](ode_linear_systems.md) §2.2 ($m = 1$, $c = 2$, $k = 5$): $G(s) = \dfrac{1}{s^2 + 2s + 5}$, so $|G(i\omega)| = \dfrac{1}{\sqrt{(5 - \omega^2)^2 + 4\omega^2}}$.

- DC gain $G(0) = 1/5$ ($-14$ dB) — the static compliance $1/k$.
- $\omega_n = \sqrt5$, $\zeta = 1/\sqrt5$, so $M_r = \dfrac{1}{2 \cdot 0.447 \cdot 0.894} = 1.25$ at $\omega_r = \sqrt5\sqrt{1 - 0.4} = \sqrt3$.
- Direct check at $\omega^2 = 3$: $(5 - 3)^2 + 4 \cdot 3 = 16$, so $|G| = 1/4 = 1.25 \times \tfrac15$ ✓.
- Far above resonance the gain falls at $-40$ dB/decade: the mass is too heavy to follow fast forcing.

> **Why delay is dangerous:** a pure delay leaves gain unchanged but adds phase lag $\omega T_d$ that grows without bound. In a feedback loop, lag at the frequency where loop gain crosses $0$ dB eats the **phase margin** and eventually causes oscillation. Sensor latency, filtering (§8.5), and sample-and-hold (§5.4) all add delay.

### 4.4 Bandwidth & Speed

The **bandwidth** $\omega_B$ is the frequency where the gain drops $3$ dB below its low-frequency value — roughly, the fastest signal the system can follow.

| System | Bandwidth | Step rise time (10–90%) |
|--------|-----------|-------------------------|
| First-order, time constant $\tau$ | $\omega_B = 1/\tau$ | $t_r \approx 2.2\,\tau$ |
| General rule of thumb | $f_B = \omega_B / 2\pi$ | $t_r \cdot f_B \approx 0.35$ |

Bandwidth and speed are the same thing viewed in two domains (§3.5). A joint position loop with $10$ Hz bandwidth rises in about $35$ ms; a sensor filter with $5$ Hz bandwidth smears out any motion faster than about $70$ ms.

---

## 5. Sampling

### 5.1 What Sampling Does to the Spectrum

Sampling $x(t)$ every $T$ seconds keeps only $x[k] = x(kT)$. In the frequency domain, the spectrum of the sampled signal is the original spectrum **repeated** at every multiple of the sample rate $f_s$:

$$X_s(f) = f_s\sum_{n=-\infty}^{\infty} X(f - n f_s)$$

If the copies do not overlap, the original can be recovered by keeping only the central copy. If they overlap, high frequencies fold onto low ones and the information is lost.

### 5.2 The Nyquist–Shannon Theorem

> **Theorem.** If $x(t)$ contains no frequencies at or above $f_{\max}$, it is **exactly** recoverable from samples taken at $f_s > 2f_{\max}$.

| Term | Value |
|------|-------|
| **Nyquist frequency** | $f_s/2$ — the highest frequency a sampled system can represent |
| **Nyquist rate** | $2f_{\max}$ — the minimum sample rate for a given signal |

The theorem is a **floor**, not a target. Control and estimation sample 10–30× faster than the dynamics of interest ([ode_linear_systems.md](ode_linear_systems.md) §8.5); the theorem only says what is information-theoretically possible.

### 5.3 Aliasing

A component at frequency $f$ above $f_s/2$ does not disappear — it shows up at the **alias** frequency

$$f_{\text{alias}} = |f - n f_s| \quad \text{for the integer } n \text{ that brings it into } [0, f_s/2]$$

Its samples are indistinguishable from a genuine low-frequency signal, so no later processing can remove it.

**Examples at $f_s = 1000$ Hz** (Nyquist = 500 Hz):

| True frequency | Appears as | Why it matters |
|----------------|-----------|----------------|
| 100 Hz | 100 Hz | Below Nyquist — fine |
| 600 Hz | 400 Hz | Folded around 500 Hz |
| 900 Hz | 100 Hz | Looks like a real 100 Hz vibration |
| 1050 Hz | 50 Hz | Looks like mains hum |
| 20 kHz (motor PWM ripple) | **0 Hz** | Exactly a multiple of $f_s$ → appears as a constant **bias** |

The last row is a real robotics failure mode: switching noise from motor drivers, aliased by a slower sensor, shows up as offset or slow drift that no amount of averaging removes.

### 5.4 Anti-Aliasing & Reconstruction

**Anti-aliasing:** remove content above $f_s/2$ **before** sampling — with an analog low-pass filter in front of the ADC, or by choosing an IMU's built-in digital low-pass (DLPF) setting and output rate together. Filtering after sampling is too late.

**Reconstruction:** a DAC or motor driver holds each command constant for one period — a **zero-order hold**. Its frequency response

$$H_{\text{ZOH}}(i\omega) = T\,\frac{\sin(\omega T/2)}{\omega T/2}\,e^{-i\omega T/2}$$

shows two side effects: a slight gain droop toward Nyquist, and an effective **delay of $T/2$** — one reason slow control loops lose phase margin.

---

## 6. The z-Transform

### 6.1 Definition & Pairs

The $z$-transform is to discrete-time signals what the Laplace transform is to continuous ones:

$$X(z) = \mathcal{Z}\{x[k]\} = \sum_{k=0}^{\infty} x[k]\, z^{-k}, \qquad z \in \mathbb{C}$$

**Computing one pair** — $x[k] = a^k$ (a geometric series):

$$X(z) = \sum_{k=0}^{\infty}\left(\frac{a}{z}\right)^k = \frac{1}{1 - a/z} = \frac{z}{z - a} \qquad (|z| > |a|)$$

| $x[k]$, $k \geq 0$ | $X(z)$ | Pole |
|-------------------|--------|------|
| $\delta[k]$ | $1$ | — |
| $\delta[k - n]$ | $z^{-n}$ | — |
| $u[k]$ | $\dfrac{z}{z - 1}$ | $1$ |
| $a^k$ | $\dfrac{z}{z - a}$ | $a$ |
| $k\,a^k$ | $\dfrac{az}{(z - a)^2}$ | $a$ (double) |
| $a^k\cos\Omega k$ | $\dfrac{z(z - a\cos\Omega)}{z^2 - 2a\cos\Omega\, z + a^2}$ | $a\,e^{\pm i\Omega}$ |

Same reading rule as Laplace: a pole at $z = p$ means a term $p^k$ in the sequence.

### 6.2 Properties & Difference Equations

| Property | Sequence | $z$-domain |
|----------|----------|-----------|
| Linearity | $a x[k] + b y[k]$ | $aX + bY$ |
| **One-step delay** | $x[k - 1]$ | $z^{-1}X(z)$ (zero initial condition) |
| One-step advance | $x[k + 1]$ | $zX(z) - z\,x[0]$ |
| Convolution | $(h * x)[k]$ | $H(z)X(z)$ |
| Final value | $\lim_{k\to\infty} x[k]$ | $\lim_{z\to 1}(z - 1)X(z)$ — if stable |

$z^{-1}$ **is a one-sample delay** — the discrete counterpart of $\tfrac1s$ being an integrator. Block diagrams of digital filters are built from $z^{-1}$ boxes.

**Solving a difference equation.** The exponential moving average $y[k] = 0.9\,y[k-1] + 0.1\,x[k]$:

$$Y = 0.9\,z^{-1}Y + 0.1\,X \quad\Rightarrow\quad H(z) = \frac{Y}{X} = \frac{0.1}{1 - 0.9z^{-1}} = \frac{0.1\,z}{z - 0.9}$$

**Step response** ($X = \tfrac{z}{z - 1}$): expand $Y(z)/z$ in partial fractions (dividing by $z$ first makes the result match the table's $\tfrac{z}{z - a}$ form):

$$\frac{Y(z)}{z} = \frac{0.1\,z}{(z - 0.9)(z - 1)} = \frac{1}{z - 1} - \frac{0.9}{z - 0.9} \quad\Rightarrow\quad y[k] = 1 - 0.9\cdot 0.9^k = 1 - 0.9^{k+1}$$

Check: $y[0] = 0.1$, $y[1] = 0.9 \cdot 0.1 + 0.1 = 0.19 = 1 - 0.81$ ✓. Final value $1$ — unit DC gain.

### 6.3 Discrete Transfer Functions & Poles

Any linear difference equation gives a rational $H(z)$. From a discrete state-space model $\mathbf{x}_{k+1} = A_d\mathbf{x}_k + B_d\mathbf{u}_k$, $\mathbf{y}_k = C\mathbf{x}_k + D\mathbf{u}_k$:

$$H(z) = C(zI - A_d)^{-1}B_d + D$$

so the poles are the eigenvalues of $A_d$. **Stable iff all poles satisfy $|z| < 1$.**

| Pole location in the $z$-plane | Behavior of $p^k$ |
|--------------------------------|------------------|
| $0 < p < 1$ | Smooth decay; slower as $p \to 1$ |
| $-1 < p < 0$ | Decay with sign flipping every sample (ringing at Nyquist) |
| $p = 1$ | Constant — a discrete integrator (accumulator) |
| $\lvert p \rvert > 1$ | Grows — unstable |
| Complex pair $r\,e^{\pm i\Omega}$ | Oscillation at $\Omega$ rad/sample, envelope $r^k$ |
| $p = 0$ | Pure delay — finite memory (FIR, §8.4) |

**Frequency response** of a discrete system: evaluate on the unit circle, $H(e^{i\omega T})$. It is periodic in $\omega$ with period $2\pi/T$ — again, aliasing.

### 6.4 From s to z

Exact sampling of $e^{st}$ gives $e^{skT} = (e^{sT})^k$, so continuous poles map as

$$z = e^{sT}$$

| $s$-plane | $z$-plane |
|-----------|-----------|
| Left half-plane ($\operatorname{Re}s < 0$, stable) | Inside the unit circle |
| Imaginary axis | Unit circle |
| $s = 0$ | $z = 1$ |
| $s = i\pi/T$ (Nyquist frequency) | $z = -1$ |
| Fast stable pole (far left) | Near $z = 0$ |

This is exactly what the ZOH discretization of [ode_linear_systems.md](ode_linear_systems.md) §8.3 does to eigenvalues. For $\dot{x} = -ax + bu$ the result is

$$H(z) = \frac{\tfrac{b}{a}\big(1 - e^{-aT}\big)}{z - e^{-aT}}$$

with pole $e^{-aT}$ ✓. Common $s \to z$ substitutions for turning a continuous design into code:

| Method | Substitution | Typical use |
|--------|-------------|-------------|
| ZOH (step-invariant) | Exact for piecewise-constant inputs | Plant models for control and estimation |
| Tustin (bilinear) | $s \leftarrow \dfrac{2}{T}\,\dfrac{z - 1}{z + 1}$ | Digital filters and controllers; maps stable to stable, warps frequency near Nyquist |
| Forward Euler | $s \leftarrow \dfrac{z - 1}{T}$ | Quick and simple; can turn stable into unstable |
| Backward Euler | $s \leftarrow \dfrac{z - 1}{Tz}$ | Always stable; adds damping |

---

## 7. The DFT & FFT

### 7.1 The Discrete Fourier Transform

Real data is a **finite** list of $N$ samples. The **DFT** gives its spectrum at $N$ evenly spaced frequencies:

$$X[m] = \sum_{k=0}^{N-1} x[k]\, e^{-i2\pi mk/N}, \qquad x[k] = \frac{1}{N}\sum_{m=0}^{N-1} X[m]\, e^{i2\pi mk/N}, \qquad m = 0, \ldots, N-1$$

It is a change of basis to $N$ orthogonal complex sinusoids — an $N \times N$ matrix multiplication ([linear_algebra.md](../basics/linear_algebra.md) §3.4).

**Tiny example.** $N = 4$, $x = [1, 0, -1, 0]$ (a cosine at $f_s/4$). With $e^{-i2\pi/4} = -i$:

$$X[m] = \sum_k x[k](-i)^{mk} = 1 - (-i)^{2m} \quad\Rightarrow\quad X = [0,\ 2,\ 0,\ 2]$$

Energy appears in bin $1$ (frequency $f_s/4$) and its mirror bin $3$. Amplitude $= 2|X[1]|/N = 1$ ✓.

### 7.2 Frequency Bins & Resolution

| Quantity | Formula | Example: $f_s = 1$ kHz, 2 s record |
|----------|---------|-----------------------------------|
| Number of samples | $N = f_s \times \text{duration}$ | $2000$ |
| Bin $m$ frequency | $f_m = m\,f_s / N$ | $f_m = 0.5\,m$ Hz |
| **Resolution** | $\Delta f = f_s / N = 1/\text{duration}$ | $0.5$ Hz |
| Highest frequency | $f_s/2$ (bin $N/2$) | $500$ Hz |

For real signals $X[N - m] = \overline{X[m]}$: bins above $N/2$ mirror the lower half (negative frequencies). Use the **one-sided** spectrum, bins $0$ to $N/2$.

> **Resolution comes from duration, not sample rate.** To separate two vibration modes $0.2$ Hz apart you need at least a $5$ s record, however fast you sample (§3.5).

### 7.3 The FFT Algorithm

Computing the DFT directly costs $O(N^2)$ operations. The **Fast Fourier Transform** (Cooley–Tukey) splits the sum into even- and odd-indexed samples — two DFTs of size $N/2$ — and recurses:

$$X[m] = \underbrace{\sum_{j} x[2j]\,e^{-i2\pi mj/(N/2)}}_{\text{DFT of evens}} + e^{-i2\pi m/N}\underbrace{\sum_{j} x[2j+1]\,e^{-i2\pi mj/(N/2)}}_{\text{DFT of odds}}$$

giving $O(N\log N)$. For $N = 1024$: about $10^6$ vs $10^4$ operations, roughly $100\times$ faster; the gap widens with $N$. The FFT computes **exactly** the DFT — it is an algorithm, not an approximation.

### 7.4 Practical Spectrum Analysis

**Spectral leakage.** The DFT implicitly treats the record as one period of a periodic signal. If a sinusoid does not complete an integer number of cycles, the wrap-around creates a jump, and its energy **leaks** into neighboring bins (the multiplication property of §3.3 with a rectangular window).

| Technique | What it does | Cost |
|-----------|-------------|------|
| **Window** (Hann, Hamming) | Tapers the record to zero at the ends; greatly reduces leakage | Wider peaks; amplitude must be rescaled |
| Remove the mean / trend | Prevents a huge DC bin hiding everything else | — |
| **Zero padding** | Interpolates the spectrum on a finer grid | Adds **no** real resolution |
| **Averaging** (Welch's method) | Averages spectra of overlapping segments; smooth noise estimate (PSD) | Coarser resolution per segment |

**Example — find a vibration in IMU data:**

```python
import numpy as np

fs, duration = 1000, 2.0
t = np.arange(0, duration, 1/fs)
x = 0.3*np.sin(2*np.pi*47*t) + 0.1*np.random.randn(t.size) + 0.5   # vibration + noise + bias

x = x - x.mean()                       # remove DC
w = np.hanning(x.size)                 # window against leakage
X = np.fft.rfft(x * w)                 # one-sided FFT
f = np.fft.rfftfreq(x.size, 1/fs)      # bin frequencies, 0 ... 500 Hz in 0.5 Hz steps
amp = 2*np.abs(X) / w.sum()            # amplitude scaling for a windowed sinusoid

print(f[np.argmax(amp)], amp.max())    # ~47.0 Hz, ~0.3
```

For noise rather than tones, use `scipy.signal.welch` to get a power spectral density.

---

## 8. Filters

### 8.1 Filter Types & Specifications

| Type | Passes | Robotics use |
|------|--------|-------------|
| **Low-pass** | Below cutoff $f_c$ | Remove sensor noise; smooth commands |
| **High-pass** | Above cutoff | Remove bias and slow drift |
| **Band-pass** | A band | Isolate one vibration mode |
| **Notch** (band-stop) | All except a narrow band | Suppress a structural resonance or mains hum |

| Specification | Meaning |
|---------------|---------|
| Cutoff frequency | Where gain falls $3$ dB |
| Roll-off | Attenuation slope beyond cutoff — $-20n$ dB/decade for order $n$ |
| Passband ripple | Gain variation inside the passband |
| Phase lag / delay | How late the output is — critical inside feedback loops (§8.5) |

### 8.2 First-Order Low-Pass, Continuous & Discrete

$$H(s) = \frac{1}{\tau s + 1}, \qquad \omega_c = \frac1\tau, \qquad f_c = \frac{1}{2\pi\tau}$$

Discretized (pole $e^{-T/\tau}$, §6.4), it becomes the **exponential moving average** used in countless robot codebases:

$$y[k] = \alpha\, y[k-1] + (1 - \alpha)\, x[k], \qquad \alpha = e^{-T/\tau}, \qquad H(z) = \frac{(1 - \alpha)\,z}{z - \alpha}$$

DC gain $H(1) = 1$ ✓. ($\alpha = \tau/(\tau + T)$ from backward Euler is nearly identical when $T \ll \tau$.)

**Example:** $f_s = 1$ kHz, desired $f_c = 10$ Hz → $\tau = \tfrac{1}{2\pi \cdot 10} = 15.9$ ms, $\alpha = e^{-1/15.9} = 0.939$. Low-frequency signals come out about $\tau = 15.9$ ms late.

```python
class LowPass:
    def __init__(self, fc, fs):
        self.a = np.exp(-2*np.pi*fc/fs)
        self.y = None
    def __call__(self, x):
        self.y = x if self.y is None else self.a*self.y + (1 - self.a)*x
        return self.y
```

### 8.3 Butterworth & Higher-Order Filters

The **Butterworth** filter is maximally flat in the passband:

$$|H(i\omega)|^2 = \frac{1}{1 + (\omega/\omega_c)^{2n}}$$

Every order is $-3$ dB at $\omega_c$; higher order $n$ gives a sharper transition.

| Order $n$ | Gain at $10\,\omega_c$ | Roll-off |
|-----------|----------------------|----------|
| 1 | $-20$ dB | $-20$ dB/decade |
| 2 | $-40$ dB | $-40$ dB/decade |
| 4 | $-80$ dB | $-80$ dB/decade |

The 2nd-order Butterworth is the second-order system of §4.3 with $\zeta = 1/\sqrt2$: $H(s) = \dfrac{\omega_c^2}{s^2 + \sqrt2\,\omega_c s + \omega_c^2}$.

```python
from scipy import signal
sos = signal.butter(2, 10, btype='low', fs=1000, output='sos')   # 2nd order, 10 Hz
y_realtime = signal.sosfilt(sos, x)       # causal — what a robot can run online
y_offline  = signal.sosfiltfilt(sos, x)   # forward + backward — zero phase, offline only
```

Use second-order sections (`sos`) rather than polynomial coefficients for numerical robustness.

### 8.4 FIR vs IIR

| | **FIR** (finite impulse response) | **IIR** (infinite impulse response) |
|---|---|---|
| Form | $y[k] = \sum_{j=0}^{M} b_j\, x[k - j]$ — inputs only | Uses past outputs too (feedback) |
| Poles | All at $z = 0$ — always stable | Anywhere — can be unstable |
| Phase | Can be exactly linear (constant delay) | Nonlinear phase |
| Sharpness per computation | Needs many taps | Few coefficients |
| Examples | Moving average, windowed-sinc | EMA (§8.2), Butterworth (§8.3) |

**Moving average** of $N$ samples: $H(z) = \tfrac1N\sum_{j=0}^{N-1} z^{-j}$. It delays by $(N - 1)/2$ samples and has **exact nulls** at $f_s/N$ and its multiples. At $f_s = 1$ kHz with $N = 20$, it completely removes 50 Hz, 100 Hz, 150 Hz, … — handy for rejecting a known periodic disturbance — at the cost of $9.5$ ms delay.

### 8.5 Delay: the Price of Filtering

A causal filter cannot remove noise without delaying the signal: the lower the cutoff, the more history it averages and the later its output.

| Filter | Approximate low-frequency delay |
|--------|--------------------------------|
| First-order low-pass | $\tau = 1/(2\pi f_c)$ |
| $N$-tap moving average | $(N - 1)T/2$ |
| 2nd-order Butterworth | $\approx \sqrt2/\omega_c = \sqrt2/(2\pi f_c)$ |

**Inside a feedback loop** this delay is phase lag (§4.3) and directly reduces stability margins — heavy filtering of a feedback signal can make a stable controller oscillate. Rules of thumb:

- Filter **only as much as needed**, with cutoff well above the loop bandwidth (often 5–10×).
- Prefer a model-based **estimator** (Kalman filter, §9.1) over heavy filtering when both low noise and low delay are needed.
- **Offline** (logs, plots, learning datasets), use zero-phase forward–backward filtering (`filtfilt`) — no delay, but non-causal, so never online.

---

## 9. Robotics Applications

### 9.1 Complementary Filter for IMU Attitude

Two sensors measure a tilt angle $\theta$ with **complementary** flaws:

| Sensor | Good at | Bad at |
|--------|---------|--------|
| Accelerometer ($\theta_{\text{acc}}$ from the gravity direction) | Low frequencies — no drift | High frequencies — noise, corrupted by linear acceleration |
| Gyroscope (rate $\omega$, integrated) | High frequencies — smooth, fast | Low frequencies — bias integrates into drift |

Low-pass the accelerometer, high-pass the integrated gyro, with filters that **sum to one**:

$$\hat\Theta(s) = \underbrace{\frac{1}{\tau s + 1}}_{\text{low-pass}}\Theta_{\text{acc}}(s) + \underbrace{\frac{\tau s}{\tau s + 1}}_{\text{high-pass}}\cdot\frac{\Omega(s)}{s}$$

If both sensors were perfect ($\Theta_{\text{acc}} = \Theta$, $\Omega/s = \Theta$), the output is $\Theta\big(\tfrac{1}{\tau s + 1} + \tfrac{\tau s}{\tau s + 1}\big) = \Theta$ — no distortion or delay of the true angle. The crossover frequency $1/(2\pi\tau)$ decides which sensor to trust at each frequency.

**Gyro bias** $b$ (so $\Omega = b/s$ extra) contributes $\tfrac{\tau s}{\tau s + 1}\cdot\tfrac{b}{s^2}$; by the final value theorem the error settles to $\tau b$ instead of drifting forever.

**Discrete implementation** (backward Euler of $\tau\dot{\hat\theta} + \hat\theta = \theta_{\text{acc}} + \tau\omega$):

$$\hat\theta_k = \alpha\big(\hat\theta_{k-1} + \omega_k T\big) + (1 - \alpha)\,\theta_{\text{acc},k}, \qquad \alpha = \frac{\tau}{\tau + T}$$

With $T = 10$ ms and $\tau = 0.5$ s: $\alpha = 0.98$ (the classic value), crossover $0.32$ Hz, and a gyro bias of $0.01$ rad/s leaves only $0.005$ rad ($0.29^\circ$) of error. A Kalman filter is the optimal, noise-model-driven generalization of this idea.

### 9.2 Differentiating Noisy Encoder Signals

Velocity from position by finite difference, $v_k = (\theta_k - \theta_{k-1})/T$, is a differentiator: gain $\propto \omega$ (§3.3). Noise at $200$ Hz is amplified $200\times$ more than a $1$ Hz motion.

**Quantization makes it worse.** A 4096-count encoder has resolution $2\pi/4096 \approx 1.53$ mrad. At $1$ kHz a single-count change reads as a velocity jump of $1.53$ rad/s — even when the joint barely moves.

**Fixes** (all trade noise against delay):

| Method | Transfer function / idea |
|--------|-------------------------|
| Filtered derivative | $\dfrac{s}{\tau s + 1}$ — differentiate, then first-order low-pass (also the standard form of the D-term in PID) |
| Longer baseline | $(\theta_k - \theta_{k-n})/(nT)$ — less noise, more delay |
| Count timing | Measure time between counts at low speed |
| State observer / Kalman filter | Use the dynamics model; best noise–delay tradeoff |

### 9.3 Vibration Analysis & Notch Filters

Mount an accelerometer on a link, excite it (step command or tap), and take the FFT (§7.4). A sharp peak marks a **structural resonance** — a lightly damped second-order mode (§4.3).

Two standard responses:

1. **Keep the control bandwidth well below the resonance**, so the controller does not excite it.
2. **Notch it out** of the feedback signal or command:

$$H_{\text{notch}}(s) = \frac{s^2 + \omega_0^2}{s^2 + \tfrac{\omega_0}{Q}s + \omega_0^2}$$

Gain is exactly $0$ at $\omega_0$ and close to $1$ away from it; the quality factor $Q$ sets the notch width (higher $Q$ = narrower). Discretize with Tustin (§6.4) — or `scipy.signal.iirnotch` — taking care that the resonance frequency is well below Nyquist.

### 9.4 Measuring a Frequency Response

When a model is unknown, **measure** $H(i\omega)$:

1. Inject sinusoids one frequency at a time — or a **chirp** (frequency sweep) or broadband noise — into the motor command.
2. Record input $u$ and output $y$ (e.g., joint velocity).
3. Estimate $H(i\omega) \approx \dfrac{S_{yu}(\omega)}{S_{uu}(\omega)}$ (cross-spectrum over input spectrum; more robust to output noise than $Y/U$).
4. Plot the empirical Bode diagram and fit a model: the low-frequency slope and level give the gain and integrators; a resonant peak gives $\omega_n \approx \omega_r$ and $\zeta$ from $M_r$ (§4.3); the phase roll-off beyond what the poles explain reveals **delay**.

The fitted transfer function becomes a state-space model ([ode_linear_systems.md](ode_linear_systems.md) §5) for simulation, estimation, and controller tuning — closing the loop between data and the models in the companion document.

---

## Summary — What to Master First

| Priority | Topic | Because |
|----------|-------|---------|
| 1 | Frequency response and Bode plots (§4) | Reading gain, phase, bandwidth, and resonance — the language of control tuning |
| 2 | Sampling, Nyquist, aliasing (§5) | Every sensor and controller is sampled; aliasing errors cannot be undone |
| 3 | First-order low-pass and filter delay (§8.2, §8.5) | The most-used filter in robotics, and its main hidden cost |
| 4 | FFT practice: bins, resolution, windowing (§7) | Diagnosing vibration and noise from real logs |
| 5 | $z$-transform and $z = e^{sT}$ (§6) | Analyzing the discrete filters and controllers that actually run |
| 6 | Convolution ↔ multiplication, Fourier pairs (§1.3, §3) | The conceptual core that ties it all together |

**Companion documents:** [ode_linear_systems.md](ode_linear_systems.md) (Laplace transform, state space, stability, discretization) · [linear_algebra.md](../basics/linear_algebra.md) (orthogonal projections and change of basis behind Fourier analysis) · [statistics_probability.md](../basics/statistics_probability.md) (noise models and Gaussians for Kalman filtering)
