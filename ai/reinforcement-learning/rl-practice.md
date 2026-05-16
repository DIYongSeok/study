# RL Practice: Teaching an Agent to Solve a Maze

This document traces the complete training process of a reinforcement learning agent learning to navigate a maze — from a blank Q-table to a reliable optimal policy. Every step is shown concretely: which cells the agent visits, how Q-values change, and what the policy looks like at each stage.

---

## Table of Contents

1. [The Maze](#1-the-maze)
2. [MDP Formulation](#2-mdp-formulation)
3. [Algorithm: Q-Learning](#3-algorithm-q-learning)
4. [Before Training: The Empty Q-Table](#4-before-training-the-empty-q-table)
5. [Phase 1 — Random Wandering (Episodes 1–5)](#5-phase-1--random-wandering-episodes-15)
6. [Phase 2 — First Goal Discovery (Episodes 6–20)](#6-phase-2--first-goal-discovery-episodes-620)
7. [Phase 3 — Value Propagation (Episodes 20–80)](#7-phase-3--value-propagation-episodes-2080)
8. [Phase 4 — Policy Emerges (Episodes 80–200)](#8-phase-4--policy-emerges-episodes-80200)
9. [Phase 5 — Convergence (Episodes 200+)](#9-phase-5--convergence-episodes-200)
10. [The Final Learned Policy](#10-the-final-learned-policy)
11. [What to Monitor During Training](#11-what-to-monitor-during-training)

---

## 1. The Maze

### Layout

```
  Col: 0  1  2  3  4  5
Row 0: #  #  #  #  #  #
Row 1: #  A  B  C  D  #
Row 2: #  #  E  #  F  #
Row 3: #  G  H  I  J  #
Row 4: #  K  #  L  M  #
Row 5: #  #  #  #  #  #
```

- `#` = wall (impassable)
- `A` = start position
- `L` = goal position
- All other letters = free cells the agent can occupy

### Cell Reference

| Cell | Position (row, col) | Role |
|------|---------------------|------|
| A | (1, 1) | **Start** |
| B | (1, 2) | — |
| C | (1, 3) | — |
| D | (1, 4) | — |
| E | (2, 2) | — |
| F | (2, 4) | — |
| G | (3, 1) | — |
| H | (3, 2) | — |
| I | (3, 3) | — |
| J | (3, 4) | — |
| K | (4, 1) | Dead end |
| L | (4, 3) | **Goal** |
| M | (4, 4) | — |

### Adjacency Map

Each cell's reachable neighbors by action:

| Cell | Up | Down | Left | Right |
|------|----|------|------|-------|
| A | wall | wall | wall | B |
| B | wall | E | A | C |
| C | wall | wall | B | D |
| D | wall | F | C | wall |
| E | B | H | wall | wall |
| F | D | J | wall | wall |
| G | wall | K | wall | H |
| H | E | wall | G | I |
| I | wall | **L** | H | J |
| J | F | M | I | wall |
| K | G | wall | wall | wall |
| L | — | — | — | — | ← terminal
| M | J | wall | L | wall |

### Possible Paths from A to L

| Path | Route | Steps |
|------|-------|-------|
| **Shortest** | A → B → E → H → I → L | **5** |
| Long route | A → B → C → D → F → J → I → L | 7 |
| Via M | A → B → C → D → F → J → M → L | 7 |
| Dead end then back | A → B → E → H → G → K → G → H → I → L | 9+ |

The agent doesn't know any of this at the start. It has to discover these paths through exploration.

---

## 2. MDP Formulation

### State Space

Each state is the agent's current cell. There are **13 possible states**: A, B, C, D, E, F, G, H, I, J, K, M, and L (terminal).

### Action Space

At every step the agent chooses one of four actions: **Up, Down, Left, Right**.

If an action leads into a wall, the agent **stays in place** and still pays the step penalty.

### Reward Function

| Situation | Reward |
|-----------|--------|
| Reach goal cell L | **+100** |
| Each step taken (including into a wall) | **−1** |

The −1 per step is crucial: it pressures the agent to find shorter paths rather than wandering forever. Without it, the agent would be indifferent between a 5-step path and a 100-step path.

### Transition Function

The maze is **deterministic**: if the agent is at H and chooses Right, it always arrives at I. There is no randomness in the transitions. (Real-world environments are stochastic, but a deterministic maze is the cleanest place to learn the concepts.)

### Episode

One episode = one attempt to navigate from A to L, or until a maximum step limit (e.g., 200 steps) is reached. When the agent reaches L, the episode ends and a new one begins — always starting at A.

---

## 3. Algorithm: Q-Learning

### What Q-Learning Does

Q-learning maintains a **Q-table**: a lookup table with one row per state and one column per action. The entry Q(s, a) stores the agent's current best estimate of the total discounted reward it expects if it takes action $a$ in state $s$, and then plays optimally from that point onward.

The agent uses this table to decide what to do, and then updates the table after every step based on what actually happened.

### The Update Rule

After every step — take action $a$ in state $s$, receive reward $r$, arrive at $s'$ — the Q-table is updated:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a) \right]$$

| Term | Value Used Here | Meaning |
|------|----------------|---------|
| $\alpha$ | 0.5 | Learning rate — how much to shift toward the new estimate |
| $\gamma$ | 0.9 | Discount factor — how much future rewards count |
| $r$ | −1 or +100 | Reward just received |
| $\max_{a'} Q(s', a')$ | Best Q-value in the new state | What the agent currently thinks it can earn from $s'$ onward |
| $r + \gamma \max Q(s',a') - Q(s,a)$ | **TD error** | How wrong the current estimate was — positive means it was underestimated |

**In plain English:** after each step, the agent asks: "I estimated $Q(s,a)$, but what I actually got was $r$ plus the best future value I now see. How wrong was I?" It then nudges the estimate in that direction by $\alpha$.

### Exploration: ε-Greedy

The agent can't only exploit what it already knows — early on the Q-table is empty, so it has to explore randomly.

**ε-greedy rule:** at each step, with probability ε choose a **random** action; with probability (1 − ε) choose the action with the **highest Q-value** for the current state.

| Training Stage | ε | Behavior |
|---|---|---|
| Episode 1 | 1.0 | 100% random — pure exploration |
| Episode 50 | 0.5 | Half random, half best-known |
| Episode 150 | 0.1 | 90% exploit, 10% explore |
| Episode 300+ | 0.05 | Almost entirely greedy |

ε decays over training because early on the Q-table has nothing useful, so random exploration is the only way to discover the goal. Later, the Q-table holds real knowledge and the agent should mostly use it.

---

## 4. Before Training: The Empty Q-Table

At the start, every Q-value is initialized to zero. The agent knows nothing.

**Q-table (all zeros), shown for the cells on the optimal path:**

| State | Q(Up) | Q(Down) | Q(Left) | Q(Right) |
|-------|-------|---------|---------|---------|
| A | 0 | 0 | 0 | 0 |
| B | 0 | 0 | 0 | 0 |
| E | 0 | 0 | 0 | 0 |
| H | 0 | 0 | 0 | 0 |
| I | 0 | 0 | 0 | 0 |

When all Q-values are 0, the agent cannot distinguish between actions. With ε = 1.0, it picks actions uniformly at random. The agent has no preference for any direction, any cell, or any path. It is essentially a random walker.

---

## 5. Phase 1 — Random Wandering (Episodes 1–5)

### What Happens

With ε = 1.0, the agent picks random actions. Most episodes end by hitting the step limit (200 steps) without reaching the goal. The maze is small, but random walks are very inefficient — the agent will bump into walls repeatedly, revisit cells, and wander in circles.

### Example: Episode 1

| Step | State | Action | Result | Reward |
|------|-------|--------|--------|--------|
| 1 | A | Right | → B | −1 |
| 2 | B | Up | → wall, stays B | −1 |
| 3 | B | Up | → wall, stays B | −1 |
| 4 | B | Left | → A | −1 |
| 5 | A | Right | → B | −1 |
| 6 | B | Down | → E | −1 |
| 7 | E | Down | → H | −1 |
| 8 | H | Left | → G | −1 |
| 9 | G | Down | → K | −1 |
| 10 | K | Up | → G | −1 |
| 11 | G | Right | → H | −1 |
| 12 | H | Right | → I | −1 |
| 13 | I | Left | → H | −1 |
| 14 | H | Up | → E | −1 |
| … | … | (continues wandering) | … | … |
| 200 | — | Step limit reached | Episode ends | Total: −200 |

The agent never reached L. Total return: −200.

### Q-Table After Episode 1

Every step triggers a Q-update. For example, step 1 (A → Right → B):

$$Q(A, \text{Right}) \leftarrow 0 + 0.5 \cdot [-1 + 0.9 \cdot \underbrace{\max Q(B, \cdot)}_{0} - 0] = 0 + 0.5 \cdot (-1) = -0.5$$

Since all Q-values in B are still 0, the update only reflects the −1 step penalty. After episode 1:

| State | Q(Up) | Q(Down) | Q(Left) | Q(Right) |
|-------|-------|---------|---------|---------|
| A | 0 | 0 | 0 | **−0.5** |
| B | **−0.5** | **−0.5** | **−0.5** | **−0.5** |
| E | **−0.5** | **−0.5** | 0 | 0 |
| H | **−0.5** | 0 | **−0.5** | **−0.5** |
| I | 0 | 0 | **−0.5** | 0 |

**Key observation:** Q-values are only slightly negative — the agent has learned "steps cost something" but has no idea where the goal is. Every action looks roughly equally bad at this stage.

---

## 6. Phase 2 — First Goal Discovery (Episodes 6–20)

### The First Time the Agent Finds the Goal

Eventually, a random walk happens to reach L. This is the moment that seeds all future learning — it's the first time a positive reward signal enters the Q-table.

Suppose in episode 8, the agent stumbles on this path:

```
A → B → E → H → G → H → I → L
```

(7 steps, with one backtrack through G)

**Step 7: the agent is at I and randomly picks Down → L. Reward: +100.**

Q-update for this step:
$$Q(I, \text{Down}) \leftarrow 0 + 0.5 \cdot [100 + 0.9 \cdot 0 - 0] = \mathbf{50}$$

L is terminal, so there's no future value. The entire +100 reward flows directly into Q(I, Down).

### Q-Table After First Goal Discovery

| State | Q(Up) | Q(Down) | Q(Left) | Q(Right) |
|-------|-------|---------|---------|---------|
| A | ≈ −0.5 | 0 | 0 | ≈ −0.5 |
| B | ≈ −0.5 | ≈ −0.5 | ≈ −0.5 | ≈ −0.5 |
| E | ≈ −0.5 | ≈ −0.5 | 0 | 0 |
| H | ≈ −0.5 | 0 | ≈ −0.5 | ≈ −0.5 |
| **I** | 0 | **50** | ≈ −0.5 | ≈ −0.5 |

Only Q(I, Down) is positive. Everything else is near zero or slightly negative. The agent has learned one crucial fact: **going down from I leads to +100 reward**.

### What This Means for Behavior

From this point on, whenever the agent happens to be at I (by random exploration), it will start to prefer going Down. But only if ε is low enough to let it exploit. Since ε is still around 0.9 at episode 8, the agent mostly still acts randomly.

The key change is in what happens in future episodes when the agent visits I's neighbors.

---

## 7. Phase 3 — Value Propagation (Episodes 20–80)

### How Values Spread Backward

After the goal is discovered, the positive signal at Q(I, Down) = 50 begins to propagate **backward** through the Q-table — one step per episode on average. This is how RL solves the credit assignment problem: good outcomes slowly teach the preceding states that they were good too.

### Propagation Step 1: I → H

In a later episode, the agent is at H and picks Right → I. It then picks Down → L (because Q(I, Down) = 50 is now the highest Q-value for I).

Q-update for (H → Right → I):
$$Q(H, \text{Right}) \leftarrow 0 + 0.5 \cdot [-1 + 0.9 \cdot \underbrace{\max Q(I, \cdot)}_{50} - 0] = 0 + 0.5 \cdot 44 = \mathbf{22}$$

### Propagation Step 2: H → E

Later, the agent is at E, picks Down → H, then goes Right → I → Down → L.

Q-update for (E → Down → H):
$$Q(E, \text{Down}) \leftarrow 0 + 0.5 \cdot [-1 + 0.9 \cdot \underbrace{\max Q(H, \cdot)}_{22} - 0] = 0 + 0.5 \cdot 18.8 = \mathbf{9.4}$$

### Propagation Step 3: E → B

$$Q(B, \text{Down}) \leftarrow 0 + 0.5 \cdot [-1 + 0.9 \cdot 9.4 - 0] = 0 + 0.5 \cdot 7.5 = \mathbf{3.7}$$

### Propagation Step 4: B → A

$$Q(A, \text{Right}) \leftarrow -0.5 + 0.5 \cdot [-1 + 0.9 \cdot 3.7 - (-0.5)] = -0.5 + 0.5 \cdot 2.83 = \mathbf{-0.08}$$

### Q-Table at Episode ~50

After 50 episodes, the optimal path has been traversed enough times for values to propagate from L all the way back toward A:

| State | Q(Up) | Q(Down) | Q(Left) | Q(Right) | Best Action |
|-------|-------|---------|---------|---------|-------------|
| A | ≈ −1 | ≈ −1 | ≈ −1 | **≈ 14** | Right ✓ |
| B | ≈ −1 | **≈ 33** | ≈ −1 | ≈ 10 | Down ✓ |
| E | ≈ −1 | **≈ 35** | ≈ −1 | ≈ −1 | Down ✓ |
| H | ≈ −1 | ≈ −1 | ≈ −1 | **≈ 40** | Right ✓ |
| I | ≈ −1 | **≈ 50** | ≈ −1 | ≈ 35 | Down ✓ |

The values are not fully converged yet, but the optimal path is already visible: A→Right, B→Down, E→Down, H→Right, I→Down.

**Why do values shrink as you move away from the goal?**  
Because $\gamma = 0.9$ discounts each step. Each cell one step further from the goal sees 90% of the next cell's value, minus 1 for the step cost. The signal attenuates the further you are from the goal.

### The Long Path Also Gets Values

The 7-step path (A → B → C → D → F → J → I → L) also accumulates Q-values during exploration, but they converge to lower numbers:

| State on long path | Best Q-value (converging) |
|---|---|
| J → Left (→I) | ≈ 40 (same as H→Right, both one step from I) |
| F → Down (→J) | ≈ 35 |
| D → Down (→F) | ≈ 30 |
| C → Right (→D) | ≈ 26 |
| B → Right (→C) | ≈ 22 |

Compare: **B → Down (short path) ≈ 33** vs **B → Right (long path) ≈ 22**. By episode 50, the agent is starting to prefer the shorter path from B.

---

## 8. Phase 4 — Policy Emerges (Episodes 80–200)

### ε Decreases, Exploitation Increases

By episode 80, ε has decayed to around 0.2. The agent now exploits the Q-table 80% of the time. Most episodes find the goal. The agent is no longer a random walker — it has a real policy.

### Example: Episode 100

| Step | State | ε-decision | Action | Result | Reward |
|------|-------|-----------|--------|--------|--------|
| 1 | A | exploit | Right | → B | −1 |
| 2 | B | exploit | Down | → E | −1 |
| 3 | E | explore (random) | Up | → B | −1 |
| 4 | B | exploit | Down | → E | −1 |
| 5 | E | exploit | Down | → H | −1 |
| 6 | H | exploit | Right | → I | −1 |
| 7 | I | exploit | Down | → **L** | **+100** |

**Episode return: −6 steps + 100 = +94 (effectively 7 steps due to one random detour)**

The agent mostly follows the optimal path but occasionally deviates due to exploration. These deviations are important — they still update Q-values and help the agent discover that the detour to B at step 3 is inefficient.

### Q-Values Continue Refining

Each episode adds more samples. The key refinements in this phase:

1. **Dead ends learn low values:** K is surrounded by walls on three sides. The agent visits K, gets stuck, pays many −1 penalties, and Q(G, Down) becomes clearly negative. The policy learns to avoid K.

2. **Long path gets outcompeted:** Q(B, Right) keeps getting updated when the agent goes A→B→C→D→... and arrives at the goal in 7 steps. Meanwhile Q(B, Down) gets updated when the agent goes A→B→E→H→I→L in 5 steps. The 5-step path has a higher return, so Q(B, Down) converges higher than Q(B, Right). The policy at B solidifies to Down.

3. **Multiple updates per cell:** cells on the optimal path like B and E are visited in almost every episode, receiving many Q-updates and converging faster than peripheral cells like C or K.

---

## 9. Phase 5 — Convergence (Episodes 200+)

### Fully Converged Q-Values (Optimal Path)

After enough episodes, the Q-table converges. For the optimal 5-step path, the converged values are:

**Derivation (working backward from goal):**

| Step from Goal | Cell | Action | Converged Q-value |
|---|---|---|---|
| 1 step | I | Down | $100$ |
| 2 steps | H | Right | $-1 + 0.9 \times 100 = 89$ |
| 3 steps | E | Down | $-1 + 0.9 \times 89 = 79.1$ |
| 4 steps | B | Down | $-1 + 0.9 \times 79.1 = 70.2$ |
| 5 steps | A | Right | $-1 + 0.9 \times 70.2 = 62.2$ |

**Converged Q-values (long path, for comparison):**

| Step from Goal | Cell | Action | Converged Q-value |
|---|---|---|---|
| 1 step | J | Left | $89$ (same, both reach I) |
| 2 steps | F | Down | $79.1$ |
| 3 steps | D | Down | $70.2$ |
| 4 steps | C | Right | $62.2$ |
| 5 steps | B | Right | $-1 + 0.9 \times 62.2 = 55.0$ |

At B, the agent must choose: Down (Q = 70.2) or Right (Q = 55.0). The short path wins clearly.

### Full Converged Q-Table

| State | Q(Up) | Q(Down) | Q(Left) | Q(Right) | Best Action |
|-------|-------|---------|---------|---------|-------------|
| A | −∞ | −∞ | −∞ | **62.2** | Right |
| B | −∞ | **70.2** | 62.2 | 55.0 | Down |
| C | −∞ | −∞ | 55.0 | **62.2** | Right |
| D | −∞ | **70.2** | 62.2 | −∞ | Down |
| E | **79.1** | 79.1* | −∞ | −∞ | Down |
| F | **79.1** | **79.1** | −∞ | −∞ | Down |
| G | −∞ | low | −∞ | **low** | neither useful |
| H | **89** | −∞ | low | 89* | Right (or Up) |
| I | −∞ | **100** | 89 | 89 | Down |
| J | **79.1** | 69.1 | **89** | −∞ | Left |
| K | low | −∞ | −∞ | −∞ | Up (only escape) |
| M | **79.1** | −∞ | **89** | −∞ | Left → goal |

> −∞ means "wall, action is never taken so Q stays at initial value ≈ −∞ from step penalties"  
> \* E→Up and H→Right both reach valid cells with similar values

---

## 10. The Final Learned Policy

The **policy** is the action with the highest Q-value at each cell. Displayed as arrows on the maze:

```
  Col: 0  1  2  3  4  5
Row 0: #  #  #  #  #  #
Row 1: #  →  ↓  →  ↓  #
Row 2: #  #  ↓  #  ↓  #
Row 3: #  ↓  →  ↓  ←  #
Row 4: #  ↑  #  G  ←  #
Row 5: #  #  #  #  #  #
```

### Reading the Policy

Starting from A (row 1, col 1):

```
A → (right) → B → (down) → E → (down) → H → (right) → I → (down) → L ✓
```

5 steps. Optimal.

### What About Non-Optimal Cells?

The policy still specifies an action for every cell, even ones not on the optimal path:

- **C → Right:** leads toward D, which leads down to F, which connects to I via J. The 7-step path — not optimal, but still reaches the goal.
- **K → Up:** K is a dead end. The only escape is Up back to G. The agent learned this by exhausting the other three directions (all walls).
- **M → Left:** leads directly to the goal. Even though the agent rarely visits M in the converged policy, it correctly learned that M→Left is the best move.
- **G → Right (→ H):** if the agent ever ends up at G (possible via K or H), the policy correctly directs it right toward H and the optimal path.

### The Policy Gradient in Q-Values

Notice how Q-values decrease the further you are from the goal along the optimal path:

```
Goal(100) ← I(100) ← H(89) ← E(79) ← B(70) ← A(62)
```

This gradient is what the policy follows — like water flowing downhill. At every cell, the agent moves in the direction of the steepest Q-value increase. The Q-table has turned the maze into a value landscape where the goal is a peak that the agent always climbs toward.

---

## 11. What to Monitor During Training

### Episode Return

The clearest signal of progress. Defined as total reward in one episode = $(−1 \times \text{steps}) + 100$ if goal reached.

| Episode | Typical Return | Meaning |
|---------|----------------|---------|
| 1–5 | −200 (timeout) | Never reaches goal |
| 6–20 | −100 to +80 | Occasionally finds goal, long paths |
| 20–80 | +70 to +90 | Finds goal most episodes, improving path |
| 80–200 | +90 to +94 | Nearly optimal, occasional exploration detour |
| 200+ | +94 | Optimal 5-step path (return = 100 − 6 = 94) |

> Maximum possible return = 100 − 5 (steps) = **95**, if we count the final step into L as a step.

### Steps Per Episode

Directly shows whether the agent is finding shorter paths:

| Stage | Average Steps | Path Quality |
|-------|---------------|--------------|
| Phase 1 | 200 (timeout) | No path found |
| Phase 2 | 50–150 | Long, lucky paths |
| Phase 3 | 10–30 | Approaching optimal, with detours |
| Phase 4 | 5–10 | Near-optimal most episodes |
| Phase 5 | 5 | Optimal every episode |

### Q-Value of the Start State

Track `max Q(A, *)` — the highest Q-value at the starting cell. This represents the agent's estimate of total expected reward from episode start:

| Episode | max Q(A, *) | Meaning |
|---------|-------------|---------|
| 0 | 0 | No knowledge |
| 10 | ≈ 5 | Barely connected to the goal |
| 50 | ≈ 30 | Partial path known |
| 150 | ≈ 55 | Near-converged |
| 300 | ≈ 62.2 | Fully converged |

When this value stops increasing, training has converged.

### TD Error (Bellman Error)

The TD error is $\delta = r + \gamma \max_{a'} Q(s', a') - Q(s, a)$ — how wrong the current Q-estimate was.

| Training Stage | Mean \|δ\| | Meaning |
|----------------|------------|---------|
| Early | Large (~50) | Q-values far from true values |
| Mid | Moderate (~10) | Rapid learning, values adjusting |
| Late | Small (~1) | Q-values nearly converged |
| Converged | ≈ 0 | Q-table consistent with Bellman equation |

When mean TD error approaches zero, every Q-value satisfies the Bellman equation — the agent's internal model of the maze is accurate.

### Goal Reach Rate

Fraction of episodes that actually reach L (vs. timing out):

| Episode | Goal Reach Rate |
|---------|----------------|
| 1–5 | 0% |
| 6–20 | ~20–40% |
| 20–80 | ~70–90% |
| 80–200 | ~95–99% |
| 200+ | ~100% |

---

## Summary: The Full Training Story

| Phase | Episodes | Key Event | What the Agent Learns |
|-------|----------|-----------|----------------------|
| Random walk | 1–5 | Nothing found | Steps cost −1 |
| First discovery | 6–20 | Reaches goal by luck | Q(I, Down) = 50 |
| Propagation | 20–80 | Values spread backward | Optimal path becomes visible in Q-table |
| Emergence | 80–200 | Policy solidifies | Agent reliably finds goal; short path beats long |
| Convergence | 200+ | Q-values stabilize | Optimal 5-step policy locked in |

The entire learning process comes down to one idea: **reward at the goal propagates backward through the Q-table one step per episode, guided by the Bellman equation, until every cell in the maze knows how far it is from the goal and in which direction to go.**
