# 🦿 CoCo-InEKF: State Estimation with Learned Contact Covariances in Dynamic, Contact-Rich Scenarios

> **ETH Zurich × Disney Research**  
> A hybrid proprioceptive state estimator — no contact labels, no heuristics, no retraining per robot.

---

## 📋 Executive Summary

**CoCo-InEKF** (Contact Covariance Invariant Extended Kalman Filter) is a proprioceptive state estimation framework for legged robots that tracks **linear velocity and pose** during highly dynamic, contact-rich maneuvers where traditional filters fail.

- Validated in simulation and on **Lima** — a custom 20-DoF bipedal robot running at a **600 Hz** control loop
- Achieves sub-millisecond inference while handling up to **18 concurrent contact points**
- Zero-shot generalization to unseen motions (pirouette, moonwalk)

---

## 🔥 The Core Problem

Traditional proprioceptive estimators assume contact points on the feet are **completely stationary** — treating contacts as binary states (on / off). This breaks down in:

| Failure Mode | Description |
|---|---|
| 👣 **Partial Contacts** | Heel-to-toe rolling transitions invalidate the static assumption |
| 🧊 **Directional Slippage** | Low-friction or unstable terrain causes directional sliding |
| 🤸 **Dynamic Behaviors** | Intense motions require manual expert threshold tuning or hand-labeled ground truth |

---

## 💡 The CoCo-InEKF Solution

CoCo-InEKF bridges **structured physical models** and **data-driven methods** through four innovations:

---

### 1️⃣ Fully Differentiable InEKF Architecture

Standard contact-aided InEKFs dynamically add/prune contact landmarks as contacts change — creating **discrete breaks in the computation graph**.

CoCo-InEKF instead:
- **Permanently maintains all contact candidate positions** in the filter state
- Bypasses discrete graph reconfigurations → all matrix multiplications, inversions, and manifold operations remain **fully differentiable**
- Strictly preserves the filter's core **mathematical invariance and observability** properties

---

### 2️⃣ Continuous Contact Velocity Covariances

Rather than predicting a binary contact flag, a lightweight **MLP** maps proprioceptive history into a continuous noise profile:

```
Inputs:  IMU readings, joint states, torques, relative kinematics
   ↓
  MLP
   ↓
Output: 6 elements of lower-triangular L
   ↓
Covariance:  ᴮΣ_Cᵢ = LL⊤  (symmetric, positive-semidefinite 3×3)
```

This lets the filter **smoothly modulate trust** in each contact point across the full spectrum:

> `Firm static hold` → `Directional sliding` → `Complete airborne`

---

### 3️⃣ End-to-End Training via BPTT

| Aspect | Detail |
|---|---|
| **Method** | Backpropagation Through Time (BPTT) |
| **Loss** | $L_2$ state-error on body-frame linear velocity vs. simulator ground truth |
| **No labels needed** | Eliminates hand-crafted contact labels and slip-detection heuristics |
| **Learning signal** | Network organically learns covariance mappings from downstream tracking performance |

---

### 4️⃣ Automated Contact Candidate Selection

- Uses **farthest point sampling** on the robot's rigid body meshes
- Accommodates non-foot interactions and varying robot morphologies
- Automated pipeline performs **on par with or better than** human-expert configurations

---

## 📊 Experimental Results

### Accuracy vs. Efficiency Frontier

| Method | Velocity Error | Trajectory Error | Inference Cost |
|---|:---:|:---:|:---:|
| Classical heuristic filter | higher | higher | low |
| Learned binary baseline | medium | medium | low |
| Transformer (end-to-end) | lower | lower | **heavy** |
| **CoCo-InEKF (ours)** | **lowest** | **lowest** | **sub-ms** ✅ |

> Scales efficiently to **18 concurrent contact points** while staying under 1 ms inference.

### 📐 Filter Consistency — NEES Test

- Metric: **Normalized Estimation Error Squared (NEES)** vs. ideal $\chi^2$ distribution
- CoCo-InEKF: closely matches theoretical confidence boundaries
- Baselines: rampant **overconfidence** (binary) or **underconfidence** (heuristic)

### 🤸 Real-World Agility on Lima (20-DoF Bipedal)

| Motion | Result |
|---|---|
| Aggressive dance sequences | ✅ Clean feedback, stable execution |
| **Pirouette** (unseen) | ✅ Zero-shot generalization |
| **Moonwalk** (unseen) | ✅ Zero-shot generalization |
| vs. External MoCap baseline | ✅ Outperforms |
