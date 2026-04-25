# Reinforcement Learning & DeepMimic

Reinforcement Learning (RL) is a learning paradigm where an **agent** learns to make good decisions by interacting with an **environment** and receiving **reward** as feedback. Unlike supervised learning — where you hand the model labeled examples — in RL the agent discovers what works through trial and error.

DeepMimic is a physics-based character animation system built on deep RL. It trains a simulated humanoid to reproduce motion capture clips by learning to match reference poses frame by frame.

---

## Table of Contents

1. [Big Picture: What is RL?](#1-big-picture-what-is-rl)
2. [MDP Foundation](#2-mdp-foundation)
   - 2.1 [Core Components](#21-core-components)
   - 2.2 [Objective Function](#22-objective-function)
3. [Value Functions](#3-value-functions)
   - 3.1 [State Value Function](#31-state-value-function)
   - 3.2 [Action Value Function](#32-action-value-function)
   - 3.3 [Bellman Equation](#33-bellman-equation)
4. [Policy](#4-policy)
   - 4.1 [Deterministic vs Stochastic](#41-deterministic-vs-stochastic)
   - 4.2 [Policy Gradient](#42-policy-gradient)
5. [RL Paradigms](#5-rl-paradigms)
   - 5.1 [Value-based](#51-value-based)
   - 5.2 [Policy-based](#52-policy-based)
   - 5.3 [Actor-Critic](#53-actor-critic)
6. [Advantage Function](#6-advantage-function)
7. [Exploration vs Exploitation](#7-exploration-vs-exploitation)
8. [How Training Works](#8-how-training-works)
   - 8.1 [The Training Loop](#81-the-training-loop)
   - 8.2 [On-Policy vs Off-Policy](#82-on-policy-vs-off-policy)
   - 8.3 [What Convergence Looks Like](#83-what-convergence-looks-like)
   - 8.4 [Hyperparameters That Matter](#84-hyperparameters-that-matter)
9. [Key Algorithms](#9-key-algorithms)
   - 9.1 [DQN](#91-deep-q-network-dqn)
   - 9.2 [PPO](#92-proximal-policy-optimization-ppo)
   - 9.3 [SAC](#93-soft-actor-critic-sac)
   - 9.4 [TD3](#94-twin-delayed-ddpg-td3)
10. [Training Stability](#10-training-stability)
    - 10.1 [Experience Replay](#101-experience-replay)
    - 10.2 [Target Network](#102-target-network)
    - 10.3 [Normalization](#103-normalization)
    - 10.4 [Clipping](#104-clipping)
11. [Reward Engineering](#11-reward-engineering)
    - 11.1 [Sparse vs Dense Reward](#111-sparse-vs-dense-reward)
    - 11.2 [Reward Shaping](#112-reward-shaping)
12. [DeepMimic](#12-deepmimic)
    - 12.1 [Overview](#121-overview)
    - 12.2 [MDP Formulation](#122-mdp-formulation)
    - 12.3 [Imitation Reward](#123-imitation-reward)
    - 12.4 [Reference State Initialization](#124-reference-state-initialization-rsi)
    - 12.5 [Early Termination](#125-early-termination)
    - 12.6 [Network Architecture](#126-network-architecture)
    - 12.7 [Training with PPO](#127-training-with-ppo)

---

## 1. Big Picture: What is RL?

### The Analogy

Think of training a dog. You don't explain in words what "sit" means. Instead:

- You say "sit" and wait for the dog to do something.
- If it sits, you give it a treat (+1 reward).
- If it doesn't, no treat (0 reward).
- After many repetitions, the dog learns that the action "sit-when-commanded" leads to treats.

RL works exactly the same way. The **agent** (dog) takes **actions** in an **environment** (the room), receives a **reward** (treat or no treat), and over time learns a **policy** — a mapping from situations to actions that maximizes cumulative reward.

---

### The Core Loop

Every RL training session consists of this loop repeating thousands or millions of times:

```
1. Observe current state s
2. Choose action a (based on current policy)
3. Environment transitions to new state s'
4. Receive reward r
5. Learn from (s, a, r, s') — update the policy or value estimates
6. Go back to step 1
```

---

### A Concrete Training Example: CartPole

CartPole is the classic beginner RL problem. A pole is balanced on a cart that can slide left or right. The goal: keep the pole upright as long as possible.

**State:** cart position, cart velocity, pole angle, pole angular velocity  
**Actions:** push cart left or push cart right  
**Reward:** +1 for every timestep the pole stays upright  
**Episode ends:** pole tilts past 15°, or 500 timesteps reached

What training looks like over time:

| Training Stage | Agent Behavior | Typical Episode Length |
|---|---|---|
| Episodes 1–10 | Completely random actions | ~8–15 steps |
| Episodes 50–200 | Starts reacting to pole lean, but overcorrects | ~30–80 steps |
| Episodes 500–1000 | Makes smaller corrections, catches itself | ~150–300 steps |
| Episodes 2000+ | Consistently balances for the full duration | 500 steps (solved) |

At episode 1, the agent has no idea what it's doing. It pushes randomly and the pole falls. After each episode, it updates its estimates of which actions led to more reward. Gradually, it discovers that pushing right when the pole leans right keeps things upright longer — and this gets reinforced because it leads to more +1 rewards.

---

### How RL Differs from Supervised Learning

| | Supervised Learning | Reinforcement Learning |
|---|---|---|
| Training signal | Labeled examples (input → correct output) | Reward signal (scalar, delayed, often sparse) |
| Feedback timing | Immediate per sample | Often delayed — only at end of episode |
| Data source | Fixed dataset | Generated by the agent's own experience |
| Exploration needed | No | Yes — must try new actions to discover better ones |
| Key challenge | Generalization | Credit assignment + exploration |

The **credit assignment problem** is RL's core difficulty: when the agent wins a chess game after 50 moves, which of those 50 moves deserved credit for the win? RL algorithms use value functions and temporal difference learning to propagate credit backward through time.

---

## 2. MDP Foundation

### 2.1 Core Components

Reinforcement Learning is formalized as a **Markov Decision Process (MDP)**:

| Symbol | Name | Description | CartPole Example |
|--------|------|-------------|-----------------|
| `s ∈ S` | State | Current observation of the world | [cart_pos, cart_vel, pole_angle, pole_vel] |
| `a ∈ A` | Action | Decision made by the agent | 0 = push left, 1 = push right |
| `r` | Reward | Scalar feedback signal | +1 per timestep upright |
| `P(s'｜s,a)` | Transition | Probability of next state | Physics of the pole-cart system |
| `γ ∈ [0,1)` | Discount | Weight for future rewards | 0.99 — long-sighted agent |

**Markov Property:** The next state depends only on the current state and action — not the full history. Knowing the current position and velocity of the CartPole is enough to predict what happens next; you don't need to remember what happened 10 steps ago.

$$P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} \mid s_t, a_t)$$

This property is what makes RL tractable. Without it, the agent would need to remember the entire history of the interaction to make decisions.

**Another example — Atari Pong:**

| Component | Atari Pong |
|---|---|
| State | Raw pixel values of the screen (210×160×3) |
| Action | Move paddle up, down, or stay |
| Reward | +1 when opponent misses, −1 when agent misses |
| Transition | Game physics engine |
| γ | 0.99 |

---

### 2.2 Objective Function

The agent's goal is to find a policy $\pi$ that maximizes expected discounted cumulative reward:

$$\max_{\pi} \; \mathbb{E}\!\left[\sum_{t=0}^{\infty} \gamma^t \, r_t\right]$$

| Term | Role | Detail |
|------|------|--------|
| $\max_{\pi}$ | **Optimize** | Search over all possible policies and pick the best one |
| $\mathbb{E}[\,\cdot\,]$ | **Expectation** | Average over all randomness — stochastic transitions and stochastic policy |
| $\sum_{t=0}^{\infty}$ | **Cumulative** | Sum rewards over the entire future trajectory, not just the next step |
| $\gamma^t$ | **Discount** | Shrinks the weight of far-future rewards; $\gamma^t \to 0$ as $t \to \infty$, keeping the sum finite |
| $r_t$ | **Reward** | Scalar signal received from the environment at timestep $t$ |

**Why discount?** A reward of +10 right now is worth more than +10 in 100 steps for two reasons: (1) the future is uncertain — we might not even reach that state; (2) it keeps the infinite sum mathematically finite.

**Discount factor examples:**

| γ | Far-future weight | Agent style |
|---|---|---|
| 0.99 | Rewards 100 steps away have 37% weight | Long-term planning |
| 0.90 | Rewards 10 steps away have 35% weight | Medium-term |
| 0.50 | Rewards 10 steps away have 0.1% weight | Very short-sighted |

> **Intuition:** find a policy so the agent collects as much reward as possible over time, but rewards received sooner count more than those far in the future.

---

## 3. Value Functions

Value functions estimate how good a state (or state-action pair) is under a given policy. They are the foundation for both planning and learning: instead of evaluating every possible future trajectory, a value function compresses the expected long-run outcome into a single scalar.

### 3.1 State Value Function

$V^\pi(s)$ — expected return starting from state $s$ and following policy $\pi$:

$$V^{\pi}(s) = \mathbb{E}_{\pi}\!\left[\sum_{t=0}^{\infty} \gamma^t r_t \;\middle|\; s_0 = s\right]$$

| Term | Detail |
|------|--------|
| $V^{\pi}(s)$ | **State value** — how good it is to be in state $s$ under policy $\pi$ |
| $\mathbb{E}_{\pi}[\,\cdot\,]$ | Expectation taken while following $\pi$ — averaging over all trajectories the policy can produce |
| $\sum_{t=0}^{\infty} \gamma^t r_t$ | Discounted cumulative reward from this point onward |
| $\mid s_0 = s$ | Conditioning — we start the sum from state $s$ |

**Concrete example — Chess:**  
At any position in a chess game, $V^\pi(s)$ answers: "given that both sides play according to strategy $\pi$ from here, what is my expected score?" A position where your queen is under threat has a lower $V$ than a position where you're about to checkmate.

**Key properties:**
- $V^\pi(s)$ is **policy-dependent**: the same state $s$ has a different value under a bad policy vs. an optimal one.
- $V^\pi$ has no opinion about which action to take — it scores states, not decisions. You need $Q^\pi$ to compare specific actions.
- In actor-critic methods, $V^\pi(s)$ serves as a **baseline**: it tells the critic how good the current situation is on average, so the actor can judge whether a specific action was better or worse than that average.
- The **optimal state value** $V^*(s) = \max_\pi V^\pi(s)$ is the best possible expected return achievable from $s$ under any policy.

---

### 3.2 Action Value Function

$Q^\pi(s,a)$ — expected return starting from state $s$, taking action $a$, then following $\pi$:

$$Q^{\pi}(s,a) = \mathbb{E}_{\pi}\!\left[\sum_{t=0}^{\infty} \gamma^t r_t \;\middle|\; s_0 = s,\; a_0 = a\right]$$

| Term | Detail |
|------|--------|
| $Q^{\pi}(s,a)$ | **Action value** — how good it is to take action $a$ in state $s$, then follow $\pi$ thereafter |
| $\mid s_0 = s,\; a_0 = a$ | First action is fixed to $a$; from $t=1$ onward the policy $\pi$ takes over |
| Relationship to $V^\pi$ | $V^\pi(s) = \mathbb{E}_{a \sim \pi}[Q^\pi(s,a)]$ — state value is the average of action values under $\pi$ |

**Concrete example — CartPole:**  
Suppose the pole is tilting slightly to the right. The Q-function answers:
- $Q(s,\ \text{push-right}) = 480$ — corrects the lean, episode likely lasts much longer
- $Q(s,\ \text{push-left}) = 12$ — worsens the tilt, episode ends soon

The agent picks push-right because it has the higher Q-value.

**Key properties:**
- $Q^\pi$ takes both a state and an action as input, so it can directly compare actions in the same state: whichever $a$ maximizes $Q^\pi(s,a)$ is the locally best choice.
- For a **deterministic policy** $\mu$, $V^\pi(s) = Q^\pi(s, \mu(s))$ — the state value collapses to the single action the policy always picks.
- $Q^\pi(s,a) \geq V^\pi(s)$ when $a$ is better than average; equal for all $a$ only when the policy is already optimal.
- Value-based algorithms (e.g., DQN) learn $Q^*$ directly and extract a policy greedily: $\pi(s) = \arg\max_a Q^*(s,a)$. This works for **discrete** actions but is intractable for continuous ones.

**Relationship summary:**

$$V^\pi(s) = \sum_a \pi(a \mid s)\, Q^\pi(s,a)$$
$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

The advantage $A^\pi$ (section 6) is what you get when you subtract the $V$-baseline from $Q$.

---

### 3.3 Bellman Equation

The Bellman equation decomposes a value into two parts — immediate reward and discounted future value — creating a recursive relationship that all RL algorithms exploit.

**Bellman expectation equation** (for a given policy $\pi$):

$$V^\pi(s) = \mathbb{E}_{a \sim \pi,\, s' \sim P}\!\left[r(s,a) + \gamma\, V^\pi(s')\right]$$

**Bellman optimality equation** (for the optimal policy):

$$Q^*(s,a) = r + \gamma \, \mathbb{E}_{s'}\!\left[\max_{a'} Q^*(s', a')\right]$$

| Term | Detail |
|------|--------|
| $Q^*(s,a)$ | Optimal action value — what we want to learn |
| $r$ | Immediate reward received after taking action $a$ in state $s$ |
| $\gamma$ | Discount factor — scales down the future relative to now |
| $\mathbb{E}_{s'}[\,\cdot\,]$ | Average over all possible next states $s'$ (environment stochasticity) |
| $\max_{a'} Q^*(s', a')$ | Best possible value from the next state — assumes optimal play from $s'$ onward |

**Intuition with an analogy:**  
You want to estimate travel time from your location to the airport. The Bellman equation says: "time from here = time to the next junction + time from that junction to the airport." You don't plan the entire route at once — just look one step ahead and trust your estimate of the remaining journey. RL learns these estimates by sampling many real trips.

**Why this matters:**
- The Bellman equation is a **self-consistency condition**: a correct value function must satisfy it for every $(s,a)$ pair simultaneously.
- The gap between the left and right sides is called the **TD error**: $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$. RL algorithms minimize this gap.
- **Bootstrapping**: the right-hand side uses the current estimate of $V(s')$ to update $V(s)$. This allows learning from incomplete trajectories without waiting for the episode to end, but introduces bias when value estimates are inaccurate early in training.

> **Key idea:** the value now equals the immediate reward plus the discounted best value you can get from here. RL solves this equation iteratively by reducing the TD error.

---

## 4. Policy

A **policy** $\pi$ is the agent's decision-making rule: given a state $s$, it outputs either a specific action or a probability distribution over actions. Everything the agent learns is ultimately encoded in the policy.

### 4.1 Deterministic vs Stochastic

A **deterministic policy** maps each state directly to a single action:

$$\mu(s) = a$$

A **stochastic policy** maps each state to a probability distribution and then samples from it:

$$\pi(a \mid s) = P(\text{take action } a \text{ in state } s)$$

| | Deterministic | Stochastic |
|---|---|---|
| Output | Single action | Distribution over actions |
| Exploration | External noise (ε-greedy, Gaussian noise) | Built-in via sampling |
| Used by | TD3, DDPG | PPO, SAC |
| Best for | Stable continuous control | Problems requiring built-in exploration |

**Example — Robot arm reaching a target:**
- **Deterministic:** given the arm's joint angles, always output the exact target angle delta. Clean and efficient once trained.
- **Stochastic:** output a Gaussian distribution around the target angle. The randomness helps explore different approaches during training; the distribution narrows as confidence grows.

---

### 4.2 Policy Gradient

The policy gradient theorem gives the direction to update policy parameters $\theta$ to increase expected return:

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{\pi}\!\left[\nabla_{\theta} \log \pi_{\theta}(a \mid s) \cdot A(s,a)\right]$$

| Term | Detail |
|------|--------|
| $\nabla_{\theta} J(\theta)$ | Gradient of expected return w.r.t. policy parameters — the direction to step |
| $\nabla_{\theta} \log \pi_{\theta}(a \mid s)$ | **Score function** — how sensitive the log-probability of action $a$ is to $\theta$ |
| $A(s,a)$ | **Advantage** — scales the gradient; positive → push $a$ to be more likely, negative → less likely |
| $\mathbb{E}_{\pi}[\,\cdot\,]$ | Average over trajectories sampled from the current policy |

**Intuition:**  
After each episode, look at every action taken. If an action led to above-average reward ($A > 0$), nudge the policy to make that action more probable in the same state. If it led to below-average reward ($A < 0$), make it less probable. Repeat across millions of episodes and the policy converges toward consistently picking good actions.

> **Rule:** increase the probability of actions that were better than average ($A > 0$), decrease the probability of actions that were worse ($A < 0$).

---

## 5. RL Paradigms

Three main strategies exist for solving RL problems, differing in what they learn and how they derive behavior.

### 5.1 Value-based

Learns $Q(s,a)$ and acts greedily: $\pi(s) = \arg\max_{a}\, Q(s,a)$.

The agent doesn't directly learn a policy. Instead, it learns a quality score for every (state, action) pair, then always picks the action with the highest score.

**Example — Atari game playing (DQN):**  
The agent observes the game screen and outputs a Q-value for each button (up, down, fire, etc.). It presses whichever button has the highest Q-value. During training, Q-values are updated based on the rewards that followed each button press.

**Limitation:** With continuous actions (e.g., joint torques for a robot), $\arg\max_a Q(s,a)$ requires solving a continuous optimization problem at every single step — which is too slow.

---

### 5.2 Policy-based

Directly optimizes the policy parameters without maintaining a value function. The policy itself is a neural network, trained by gradient ascent on expected return.

**Example — REINFORCE on CartPole:**  
The policy network takes the cart/pole state as input and outputs the probability of pushing left vs. right. After each episode, if the episode was long (high return), increase the probability of every action taken during it. If it was short, decrease them. No Q-values computed at all.

**Limitation:** Very high variance — a single episode's outcome can vary wildly even under the same policy. Requires many samples to get a stable gradient signal.

---

### 5.3 Actor-Critic

Combines both: the **Actor** learns the policy; the **Critic** learns the value function to reduce gradient variance.

The actor takes actions; the critic tells the actor how good or bad those actions were relative to the current value estimate — a much more informative training signal than raw returns alone.

**Example — PPO training a humanoid to walk:**
- The **Actor** outputs a Gaussian distribution over joint torques.
- The **Critic** estimates $V(s)$ — how much reward the humanoid is expected to earn from its current body position.
- After each step: if the humanoid moved forward (positive advantage), the actor increases the probability of that torque pattern. If it fell (negative advantage), the actor decreases it.

| Method | Variance | Bias | Action Space | Examples |
|--------|----------|------|--------------|---------|
| Value-based | Low | High | Discrete | DQN |
| Policy-based | High | Low | Continuous | REINFORCE |
| Actor-Critic | Low | Low | Both | PPO, SAC, TD3 |

---

## 6. Advantage Function

The advantage $A(s,a)$ measures how much better action $a$ is compared to the **average** action in state $s$:

$$A(s,a) = Q(s,a) - V(s)$$

| Term | Detail |
|------|--------|
| $Q(s,a)$ | Value of taking this specific action $a$ |
| $V(s)$ | Baseline — average value of state $s$ regardless of action |
| $A > 0$ | Action $a$ was better than the average action in state $s$ |
| $A < 0$ | Action $a$ was worse than the average action in state $s$ |

**Why advantage instead of raw return?**  
Suppose an episode returns total reward = 200. Was that good or bad? Without a baseline, you don't know. If $V(s) = 180$, then the agent did better than expected by 20 — reinforce its actions. If $V(s) = 300$, it did much worse than expected — suppress its actions. The advantage tells you relative performance, which is a much cleaner training signal.

Using raw returns leads to high variance. **Generalized Advantage Estimation (GAE)** smooths this by blending short-term and long-term estimates:

$$A_t^{\,\text{GAE}} = \sum_{k=0}^{\infty} (\gamma\lambda)^k \,\delta_{t+k}, \qquad \delta_t = r_t + \gamma\,V(s_{t+1}) - V(s_t)$$

| Term | Detail |
|------|--------|
| $\delta_t$ | **TD error** — one-step advantage signal; how much better $s_{t+1}$ turned out to be than predicted |
| $(\gamma\lambda)^k$ | Exponential decay — nearby TD errors weighted more heavily than distant ones |
| $\lambda \in [0,1]$ | **Trace decay**: $\lambda=0$ gives pure TD (low variance, high bias); $\lambda=1$ gives full return (high variance, low bias) |

**GAE interpolates between two extremes:**
- $\lambda = 0$: judge each action by only the next step's outcome → fast but myopic (high bias)
- $\lambda = 1$: judge each action by the full episode return → accurate but noisy (high variance)
- $\lambda = 0.95$: the practical default — mostly long-term with a little smoothing

**Key insight:** $A(s,a) > 0$ → action was better than average, increase its probability. $A(s,a) < 0$ → worse than average, decrease its probability.

---

## 7. Exploration vs Exploitation

**Exploration:** trying new, uncertain actions to discover whether they might be better than what's currently known.  
**Exploitation:** using the current best-known action to collect reward right now.

This is a fundamental tension: an agent that only exploits never discovers better actions, but an agent that only explores never collects reward.

| Strategy | Mechanism | Use Case | Example |
|----------|-----------|---------|---------|
| ε-greedy | With probability ε, take a random action; otherwise take the best known | DQN, discrete | ε = 0.1: 10% random, 90% best known |
| Entropy bonus | Reward the policy for remaining uncertain | SAC, continuous | Agent is rewarded for spreading probability across actions |
| Noise injection | Add Gaussian noise to the output action | TD3, DDPG | Robot arm: intended torque + small random wiggle |
| UCB | Prefer actions/states with high uncertainty | Bandit, model-based | Try an unexplored path even if a known path is decent |

**A concrete example of why exploration matters:**  
A robot is learning to open a door. Early on, it tries pushing directly — the door moves slightly (+0.1 reward). The robot learns "push = good" and exploits this. It never discovers that turning the handle first, then pushing, gives +10 reward. ε-greedy exploration occasionally forces a random action — sometimes accidentally turning the handle — which reveals the much better strategy.

**Entropy bonus (used by SAC):**  
SAC adds $\alpha \cdot H(\pi(\cdot \mid s))$ to the reward at each step, where $H$ is the entropy (randomness) of the policy. This rewards the agent for keeping its options open. As training progresses and the agent becomes confident, the temperature $\alpha$ can be annealed so the policy becomes more decisive.

---

## 8. How Training Works

This section explains the full training loop concretely — what actually happens step by step when you train an RL agent.

### 8.1 The Training Loop

A single training iteration for an on-policy algorithm like PPO:

**Step 1 — Collect a rollout:**  
Run the current policy in the environment for $N$ steps (e.g., $N = 4096$). At each step, record $(s_t, a_t, r_t, s_{t+1})$. This produces one batch of experience called a **rollout**.

**Step 2 — Compute returns and advantages:**  
Starting from the last timestep and working backward, compute the discounted return $G_t = r_t + \gamma G_{t+1}$ for each step. Then compute advantages $A_t = G_t - V(s_t)$ using the critic's current estimates.

**Step 3 — Update the networks:**  
Use the collected batch to run several gradient steps. Update the actor to increase the probability of high-advantage actions and the critic to better predict future returns.

**Step 4 — Discard the data and repeat:**  
For on-policy methods, the batch is thrown away after updating (since the policy has changed, old data is no longer valid under the new policy). Collect a fresh rollout and repeat.

---

### 8.2 On-Policy vs Off-Policy

| | On-Policy (PPO) | Off-Policy (SAC, DQN) |
|---|---|---|
| Data freshness | Must use data from the current policy | Can reuse old data via a replay buffer |
| Sample efficiency | Lower — data is discarded each update | Higher — each transition is reused many times |
| Stability | More stable (data and policy stay aligned) | Can be unstable if old data is too stale |
| Typical use | Physics-based control, locomotion | Robotic manipulation, game playing |

**Off-policy training with a replay buffer:**  
SAC and DQN store every $(s, a, r, s')$ transition in a large memory buffer (e.g., 1 million transitions). Each update step randomly samples a mini-batch from this buffer. Because the buffer contains data from many past policy versions, the agent keeps learning from old experience — making it far more sample-efficient than on-policy methods.

---

### 8.3 What Convergence Looks Like

Training progress is tracked with these metrics:

- **Episode reward (mean return):** should trend upward over training — the clearest signal that the agent is improving.
- **Value loss:** how well the critic fits actual returns — should decrease and stabilize.
- **Policy entropy:** should start high (random policy) and gradually decrease as the policy becomes more decisive.
- **KL divergence (PPO):** how much the policy changes per update — should stay small to prevent instability.

**Example training curve — robot locomotion:**

| Training Step | Mean Episode Reward | Observed Behavior |
|---|---|---|
| 0 | 0.2 | Falls immediately, random thrashing |
| 100k | 1.5 | Can stand briefly; crude stepping motions |
| 500k | 8.0 | Slow forward movement; frequent falls |
| 2M | 25.0 | Consistent walking gait |
| 10M | 40.0 | Efficient, near-optimal walking speed |

The curve is rarely smooth — there are often dips where the policy tries a new strategy and temporarily gets worse before improving.

---

### 8.4 Hyperparameters That Matter

| Hyperparameter | What It Controls | If Too High | If Too Low |
|---|---|---|---|
| Learning rate | Step size per gradient update | Unstable, diverges | Learns too slowly |
| γ (discount) | How far-sighted the agent is | Can cause instability | Agent ignores the future (myopic) |
| N (rollout steps) | Data per update | Slow updates, high memory | Noisy gradient estimates |
| Clip ε (PPO) | Max policy change per step | Unstable updates | Learns too slowly |
| Replay buffer size | How much old data is retained | High memory, stale data | Forgets past experience quickly |

---

## 9. Key Algorithms

### 9.1 Deep Q-Network (DQN)

Uses a neural network to approximate $Q(s,a)$ for **discrete** action spaces. The key innovation was combining Q-learning with deep neural networks and two stabilization techniques: experience replay and a target network.

**How it works step by step:**
1. Observe state $s$ (e.g., game screen pixels).
2. Feed $s$ through the Q-network; get a Q-value for each possible action.
3. Pick the action with the highest Q-value, or a random action with probability ε.
4. Store $(s, a, r, s')$ in the replay buffer.
5. Sample a random mini-batch from the buffer; compute TD targets using the target network.
6. Update the Q-network to minimize $(Q(s,a) - \text{target})^2$.
7. Periodically copy Q-network weights to the target network.

**Best for:** Atari games, board games, any problem with discrete actions (buttons, directions, discrete choices).

**Limitation:** Cannot handle continuous action spaces — $\arg\max_a Q(s,a)$ over a continuous range requires solving an optimization problem at every step.

---

### 9.2 Proximal Policy Optimization (PPO)

The industry standard for stable, sample-efficient on-policy RL. Clips the policy update ratio to prevent destructive updates.

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}\!\left[\min\!\left(r_t(\theta)\,A_t,\;\operatorname{clip}\!\left(r_t(\theta),\,1{-}\varepsilon,\,1{+}\varepsilon\right)A_t\right)\right]$$

$$r_t(\theta) = \frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

| Term | Detail |
|------|--------|
| $r_t(\theta)$ | **Probability ratio** — how much more or less likely the new policy is to take the same action |
| $r_t = 1$ | New and old policy agree exactly on this action |
| $r_t > 1$ | New policy made this action more likely |
| $r_t < 1$ | New policy made this action less likely |
| $A_t$ | Advantage — determines whether we want to push the action up or down |
| $\operatorname{clip}(\cdot, 1{-}\varepsilon, 1{+}\varepsilon)$ | Hard constraint: ratio is clamped to $[0.8, 1.2]$ when $\varepsilon = 0.2$ |
| $\min(\cdot,\cdot)$ | Take the pessimistic bound — prevents the update from being too greedy in either direction |

**The clipping intuition:**  
Without clipping, a single lucky rollout could push the policy very far in one direction — destroying what it had previously learned. The clip ensures no single update changes action probabilities by more than a factor of $(1 \pm 0.2)$. If the advantage is positive but the ratio already exceeds 1.2, the gradient is cut off — no further pushing in that direction this update.

**Total PPO loss:**

$$\mathcal{L}_{\text{total}} = \mathcal{L}^{\text{CLIP}} + c_1 \cdot \mathcal{L}^{\text{value}} - c_2 \cdot H(\pi)$$

$\mathcal{L}^{\text{value}}$ trains the critic; $H(\pi)$ is an entropy bonus that discourages premature commitment to a single action.

> **Key idea:** PPO lets the policy improve, but won't allow a single update to change it so drastically that it becomes unstable.

**Best for:** Physics-based character animation, robotic locomotion, RLHF (reinforcement learning from human feedback in LLMs).

---

### 9.3 Soft Actor-Critic (SAC)

Off-policy Actor-Critic for **continuous** control. Maximizes reward plus entropy simultaneously, so exploration is built into the objective rather than added externally.

$$J(\pi) = \mathbb{E}\!\left[\sum_{t} \gamma^t \Bigl(r_t + \alpha\,\mathcal{H}\bigl(\pi(\cdot \mid s_t)\bigr)\Bigr)\right]$$

| Term | Detail |
|------|--------|
| $r_t$ | Environment reward at step $t$ |
| $\mathcal{H}(\pi(\cdot \mid s_t))$ | **Entropy** of the policy at $s_t$ — high entropy means probability spread across many actions |
| $\alpha$ | **Temperature** — controls the trade-off between reward and entropy; higher $\alpha$ = more random |
| $r_t + \alpha\,\mathcal{H}$ | SAC maximizes reward AND randomness simultaneously |

**How it differs from PPO:**
- SAC is **off-policy**: it stores all experience in a replay buffer and reuses it many times — far more sample-efficient. Ideal when environment interaction is expensive (real-robot experiments).
- SAC uses **two Q-networks** in parallel and takes the minimum, preventing overestimation of action values.
- The entropy term means the agent prefers to remain uncertain until it has strong evidence — it won't collapse to a single action prematurely.

**Best for:** Robotic manipulation, dexterous hand control, continuous control where data is expensive.

---

### 9.4 Twin Delayed DDPG (TD3)

Deterministic policy for **continuous** control. Fixes two key failure modes in earlier actor-critic methods:

1. **Overestimation bias:** two Q-networks compute values for each action; the minimum is used. This prevents the actor from exploiting inflated Q-values.
2. **Policy oscillation:** the actor is updated only every 2 critic steps. This lets the critic stabilize before the actor acts on its estimates.
3. **Target policy smoothing:** Gaussian noise is added to target actions when computing TD targets. This prevents sharp Q-function peaks around specific actions, making the value landscape smoother.

| | SAC | TD3 |
|---|---|---|
| Policy type | Stochastic | Deterministic |
| Exploration | Built-in (entropy bonus) | External noise injection |
| Twin critics | Yes | Yes |
| Entropy objective | Yes | No |

**Algorithm comparison:**

| Algorithm | Action Space | On/Off-policy | Key Feature |
|-----------|-------------|---------------|-------------|
| DQN | Discrete | Off | Experience replay + target network |
| PPO | Both | On | Clipped policy update |
| SAC | Continuous | Off | Entropy maximization |
| TD3 | Continuous | Off | Twin critics + delayed actor update |

---

## 10. Training Stability

Deep RL is notoriously unstable — small changes in hyperparameters or random seeds can produce completely different results. These techniques address the most common failure modes.

### 10.1 Experience Replay

**Problem:** If you update the network on the most recent transition only, consecutive updates are highly correlated (each step closely resembles the last), leading to biased gradient estimates and instability.

**Solution:** Store all transitions in a large buffer. Sample randomly from the entire buffer for each update. This breaks temporal correlation and ensures the gradient reflects diverse experience.

**Example:** DQN's replay buffer stores 1 million transitions. Even though the agent is currently in level 5 of a game, it simultaneously trains on transitions from levels 1, 2, 3, and 4 — preventing catastrophic forgetting of earlier-learned behaviors.

---

### 10.2 Target Network

**Problem:** The TD target $r + \gamma \max_{a'} Q(s', a')$ depends on the Q-network itself. If we update the Q-network and recompute the target simultaneously, the training target keeps moving — like trying to hit a moving bullseye. This often causes divergence.

**Solution:** Keep a separate **target network** $Q_{\text{targ}}$ that updates slowly. The main network chases a stable target rather than its own moving estimate.

**Soft update (used by SAC, TD3):**

$$\theta_{\text{targ}} \leftarrow \tau \cdot \theta + (1-\tau) \cdot \theta_{\text{targ}}, \qquad \tau \approx 0.005$$

The target network lags behind the main network. Small $\tau$ → very stable target, slow to adapt. Large $\tau$ → faster adaptation, less stable.

---

### 10.3 Normalization

**Problem:** Raw observations can have very different scales — e.g., joint angles in radians (±3) vs. velocities in m/s (±50). Neural networks train poorly on unnormalized inputs because gradients are dominated by large-scale features.

**Solution:** Maintain running statistics (mean and variance) of all observed values. Normalize inputs to zero mean and unit variance at each step.

**Reward normalization:** In long episodes, cumulative rewards can grow to hundreds or thousands. Normalizing rewards to a consistent scale prevents the value loss from being dominated by outlier magnitudes.

---

### 10.4 Clipping

Three distinct clipping operations appear throughout RL:

| Type | Where Used | What It Does |
|---|---|---|
| **PPO ratio clip** | PPO policy loss | Caps policy change per update to $[1-\varepsilon, 1+\varepsilon]$ |
| **Gradient clip** | All deep RL | Prevents exploding gradients by capping the L2 norm of the gradient vector |
| **Reward clip** | DQN (Atari) | Maps all rewards to $\{-1, 0, +1\}$ so Q-values stay on the same scale across games |

Gradient clipping is especially important in recurrent networks and long-horizon tasks where gradients can grow exponentially through time (the exploding gradient problem).

---

## 11. Reward Engineering

The reward function is the most critical design choice in RL. A poorly designed reward leads to unintended behaviors (reward hacking), while a well-designed one makes training fast and stable.

### 11.1 Sparse vs Dense Reward

**Sparse reward:** Only signals success or failure, with no guidance during the episode.

**Example:** A robot tries to pick up an object. Reward = +1 if the object is in the bin, 0 otherwise. The robot performs thousands of random movements and almost never reaches the bin — so it almost never receives a training signal. Learning is very slow.

**Dense reward:** Provides continuous feedback throughout the episode.

**Example:** Reward = negative distance to the bin at each step. The robot always receives gradient information about how close it is getting. Training is much faster.

| | Sparse | Dense |
|---|---|---|
| Learning difficulty | Hard (signal is rare) | Easier (signal at every step) |
| Risk | Requires long exploration to find any reward | Can cause reward hacking |
| Use case | Precise tasks (grasp success, win/lose) | Continuous control, locomotion |

**Reward hacking example:**  
A boat racing agent was given reward for speed and hitting checkpoints. It discovered that driving in tight circles while hitting the same checkpoint repeatedly scored higher than completing the race. The agent solved the reward function, not the intended task. This is a classic failure of dense reward design.

---

### 11.2 Reward Shaping

Adds potential-based bonuses to guide learning without changing the optimal policy.

$$r'(s,a,s') = r(s,a,s') + F(s,s'), \qquad F(s, s') = \gamma\,\Phi(s') - \Phi(s)$$

| Term | Detail |
|------|--------|
| $\Phi(s)$ | **Potential function** — a hand-designed score for state $s$ (e.g., negative distance to goal) |
| $\gamma\,\Phi(s')$ | Potential of the next state, discounted |
| $F(s, s')$ | Shaping bonus — positive when moving to a better state, negative when moving to a worse one |
| $r'$ | Shaped reward — guaranteed to have the same optimal policy as the original $r$ |

**The key guarantee:** potential-based shaping cannot change what the optimal policy is — it only makes learning faster. Adding arbitrary bonus rewards (without the $\gamma \Phi(s') - \Phi(s)$ structure) can accidentally redirect the agent to a different goal entirely.

**Example — navigation:**  
True reward: +100 for reaching the goal, 0 everywhere else.  
Potential function: $\Phi(s) = -\text{distance\_to\_goal}(s)$.  
Shaping bonus at each step: $\gamma \cdot (-d_{t+1}) - (-d_t) = d_t - \gamma d_{t+1}$. This is positive when the agent moves closer to the goal, giving a continuous signal without altering the optimal destination.

---

## 12. DeepMimic

### 12.1 Overview

**DeepMimic** (Peng et al., 2018) is a physics-based character animation system that trains a simulated character to reproduce motion capture clips using deep RL. The character learns to mimic reference poses while remaining physically plausible under a physics simulator (PyBullet or MuJoCo).

Key contributions:
- **Imitation reward** — directly compares simulated pose to the reference motion at each frame
- **Reference State Initialization (RSI)** — starts each episode at a random position in the reference clip
- **Early termination** — ends episodes when the character falls, accelerating learning

---

### 12.2 MDP Formulation

| Component | Description |
|---|---|
| **State** $s$ | Joint positions, joint velocities, end-effector positions (wrists, ankles), and phase $\varphi \in [0,1]$ indicating where in the motion clip the character currently is |
| **Action** $a$ | Target joint angles passed to a PD controller, which applies torques to drive simulated joints toward those targets |
| **Reward** $r$ | Weighted imitation reward comparing simulated pose to reference pose (detailed below) |
| **Transition** | Determined entirely by the physics simulator |

**Why include phase $\varphi$ in the state?**  
Without phase, the agent doesn't know *where* in the reference motion it should be at this moment. Two identical body positions can require completely different actions depending on whether the character is mid-swing or about to land. Including phase makes the state fully informative.

**Why use a PD controller for actions?**  
Rather than directly outputting raw joint torques, the network outputs **target joint angles**. The PD controller then applies torque proportional to the angle error and velocity error: $\tau = k_p(\theta_{\text{target}} - \theta_{\text{current}}) - k_d \dot{\theta}$. This makes the action space smoother and far easier to learn from than raw torques.

---

### 12.3 Imitation Reward

The total reward is a weighted sum of four sub-rewards:

$$r_t = w_p \cdot r^p + w_v \cdot r^v + w_e \cdot r^e + w_c \cdot r^c$$

Each sub-reward uses a Gaussian kernel — it equals 1 when the simulated character perfectly matches the reference, and decays toward 0 as the deviation grows:

$$r^p = \exp\!\left(-2\sum_{j} \|\hat{q}_j - q_j^*\|^2\right) \qquad \text{(joint angles)}$$

$$r^v = \exp\!\left(-0.1\sum_{j} \|\dot{\hat{q}}_j - \dot{q}_j^*\|^2\right) \qquad \text{(joint velocities)}$$

$$r^e = \exp\!\left(-40\sum_{e} \|\hat{p}_e - p_e^*\|^2\right) \qquad \text{(end-effector positions)}$$

$$r^c = \exp\!\left(-10\,\|\hat{p}_{\text{root}} - p_{\text{root}}^*\|^2\right) \qquad \text{(root / CoM)}$$

| Symbol | Meaning |
|--------|---------|
| $\hat{q}_j$ | Simulated joint angle for joint $j$ |
| $q_j^*$ | Reference joint angle from the motion clip |
| $\dot{\hat{q}}_j,\; \dot{q}_j^*$ | Simulated and reference joint velocities |
| $\hat{p}_e,\; p_e^*$ | Simulated and reference end-effector positions (wrists, ankles) |
| $\exp(-c \|\cdot\|^2)$ | Gaussian kernel — output is 1 at zero error, smoothly penalizes larger deviations; larger $c$ = sharper penalty |

| Term | Weight | Measures | Why this weight? |
|------|--------|---------|---------|
| $r^p$ | 0.65 | Joint angle similarity | Pose is the primary goal — the character must look right |
| $r^v$ | 0.10 | Joint velocity similarity | Ensures smooth motion, not just correct snapshots |
| $r^e$ | 0.15 | Wrist/ankle position | End-effectors are perceptually salient (hands and feet) |
| $r^c$ | 0.10 | Pelvis position/orientation | Keeps the character's overall trajectory on track |

---

### 12.4 Reference State Initialization (RSI)

**Problem:** If every episode starts at the beginning of the motion clip, the agent only gets training signal for the early part of the motion. It never learns to handle the middle or end until it has already mastered the beginning — which takes enormously long.

**Solution:** Each episode starts at a **random phase** $\varphi \sim \mathcal{U}[0,1)$ — a random position in the motion clip. The simulated character is initialized to the corresponding reference pose at that phase.

**Effect:** From the very first training step, the agent receives experience from all parts of the motion. It learns to recover from any position in the clip, not just from the start. This dramatically accelerates training.

**Example — backflip clip:**
- **Without RSI:** agent learns to stand and begin the wind-up after 1M steps. Never attempts the landing because it always fails before getting there.
- **With RSI:** agent is initialized mid-backflip in one-third of episodes, and near the landing in another third. It simultaneously learns all phases of the motion from the start.

---

### 12.5 Early Termination

If the simulated character falls (root height below threshold, e.g., 0.5 m), the episode ends immediately. This serves two purposes:

1. **Efficiency:** No point continuing an episode where the character is on the ground — the imitation reward will be near zero for the rest of the episode anyway.
2. **Learning signal:** Abrupt termination signals that the preceding actions were catastrophically bad. With GAE, this propagates backward through time and penalizes the actions that led to the fall.

**Important detail:** the critic is trained to predict $V(s_{\text{terminal}}) = 0$ at fall states. The GAE computation handles episode boundaries correctly — the advantage calculation stops at the terminal state and does not bleed into the next episode.

---

### 12.6 Network Architecture

Both actor and critic are fully connected networks with Tanh activations, which are typical for physics-based control. Tanh bounds activations, which stabilizes the PD controller targets and prevents extreme joint angle outputs.

| Network | Layer Sizes | Output |
|---|---|---|
| **Actor** | obs → 1024 → 512 → act_dim | Mean of Gaussian distribution over target joint angles |
| **Critic** | obs → 1024 → 512 → 1 | Scalar state value $V(s)$ |

The actor outputs a mean and a learnable log-standard-deviation. The standard deviation starts small (log_std initialized to −2) so that early training produces small, cautious joint movements rather than wild torques that immediately destabilize the character.

---

### 12.7 Training with PPO

DeepMimic uses PPO as the base algorithm. Key hyperparameter choices differ from standard PPO because of the physics simulation context:

| Parameter | Value | Reason |
|-----------|-------|------|
| `gamma` | 0.95 | Lower than typical — motion clips are short-horizon; distant future matters less |
| `lam` (GAE) | 0.95 | Standard — balances bias and variance well |
| `clip_eps` | 0.2 | Standard PPO clipping |
| `n_steps` | 4096+ | More steps per update for stable physics gradients |
| `n_epochs` | 10 | Reuse each rollout 10 times for efficiency |
| `lr` | 5e-5 | Much smaller than typical — fine physical control requires cautious updates |
| reward weights | (0.65, 0.1, 0.15, 0.1) | Pose-heavy — visual appearance matters most |

**Why a lower learning rate?**  
In locomotion, a policy update that overshoots (e.g., making the leg swing too far) causes the character to fall. Once fallen, recovery is hard to learn. Small learning rates prevent the policy from jumping to configurations that immediately lead to falling.

**Training progression for a walk clip:**

| Stage | Steps | Observed Behavior |
|---|---|---|
| Early | 0–50k | Falls immediately in most episodes; learns basic balance |
| Mid | 50k–500k | Stable standing; crude stepping motions appear |
| Late | 500k–2M | Consistent walking gait; smoother transitions |
| Converged | 2M+ | Faithful reproduction of the reference motion clip |
