# 🎭 AMOR: Adaptive Character Control through Multi-Objective Reinforcement Learning

> **SIGGRAPH 2025** — Disney Research × Universidade Federal do Rio Grande do Sul  
> Train once. Tune forever — **zero retraining required.**

---

## 🔥 The Core Challenge

Standard RL for physics-based character animation uses a **fixed, weighted sum of rewards** set before training. This creates three critical bottlenecks:

| Problem | Description |
|---|---|
| ⚔️ **Conflicting Objectives** | High motion-tracking accuracy directly conflicts with energy efficiency and smoothness |
| 🔁 **Tedious Iteration** | Finding the right balance requires manual weight adjustments + full policy retraining each time |
| 🌉 **Sim-to-Real Gap** | Real robots need much higher smoothness weights to suppress jitter — impossible to predict in sim |

---

## 💡 The AMOR Solution

Instead of training a policy for a single fixed reward structure, AMOR trains a **single policy conditioned on a reward weight vector** $w$ — covering the entire Pareto front of behavior trade-offs.

> **Result:** Users and algorithms can do **zero-shot, on-the-fly behavior tuning** post-training — no retraining ever needed.

---

## 🛠️ Technical Components

### 🔹 MOPPO — Multi-Objective PPO
- Extends standard **Proximal Policy Optimization** to a multi-objective setting
- Learns a **vector-valued value function** and **vector-valued advantage function**
- Updates the policy across a **multi-dimensional simplex** of weights

### 🔹 Motion Context Conditioning
The policy is conditioned on three inputs simultaneously:
1. Character's **physical state**
2. **Reward weight vector** $w$
3. **Motion context** — a VAE-compressed latent window of past + future target frames

### 🔹 Seven Conflicting Objectives

| Reward Term | Purpose |
|---|---|
| **Upper Body** $r^{up}$ | Tracks upper joint positions and character height |
| **Lower Body** $r^{lo}$ | Tracks lower body joint positions |
| **Feet** $r^{feet}$ | Tracks ankle joint placement |
| **Rigid Body** $r^{rbs}$ | Measures end-effector position and orientation |
| **Root** $r^{root}$ | Monitors base orientation of the character's root |
| **Velocity** $r^{vel}$ | Tracks linear and angular joint/root velocities |
| **Smoothness** $r^{smooth}$ | Penalizes high action rates and torques to suppress vibrations |

---

## 🤖 Hierarchical Weight Adjustment — High-Level Policy (HLP)

Beyond manual tuning, AMOR includes an **automated High-Level Policy** that acts as a real-time supervisor over the frozen low-level policy.

```
[HLP] → dynamically outputs reward weight vector w
          ↓
[Low-Level AMOR Policy] → character motion
```

- **Training signal:** Adversarial discriminator scoring how *indistinguishable* simulated transitions are from real MoCap data
- **Interpretability:** The HLP explicitly outputs weight vectors — developers can visually inspect *which* physical traits (e.g., velocity vs. foot placement) the discriminator values at each moment

---

## 📊 Experimental Results

Evaluated on a **36-DoF simulated humanoid** and a **physical 20-DoF bipedal robot**.

### ✅ Sim-to-Real Jitter Elimination
- Uniform tracking weights → **heavy joint jitter** on physical robot
- Manually scaling up the smoothness weight on-the-fly → **jitter instantly eliminated**

### 🌀 Double Pirouette Transfer
Successfully transferred a complex **double pirouette** to the real robot using time-varying weights:

| Phase | Dominant Weight |
|---|---|
| Early (spin-up) | 🔺 High **velocity** weight — build rotational speed |
| Late (landing) | 🔹 High **smoothness** weight — stabilize the landing |

### ⏱️ Massive Time Savings

| Method | Time Required |
|---|:---:|
| Manual on-robot weight tuning (with AMOR) | **~1 day** |
| Training the base multi-objective policy | **5 days** |
| Traditional iterative RL retraining (without AMOR) | **Not feasible** |

### 📉 Performance Gains (HLP vs. Uniform Baseline)

| Metric | Baseline | AMOR + HLP | Improvement |
|---|:---:|:---:|:---:|
| Joint Pose MAE (degrees) | 10.02° | **9.55°** | ▼ 4.7% |
| Discriminator Score | lower | **significantly higher** | ✅ |
