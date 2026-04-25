# Reinforcement Learning & DeepMimic

Reinforcement Learning (RL) is a learning paradigm where an agent learns to maximize cumulative reward by interacting with an environment. DeepMimic is a physics-based character animation system built on deep RL that enables simulated characters to imitate motion capture clips.

```bash
pip install gymnasium stable-baselines3 torch
```

```python
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
```

## Table of Contents

1. [MDP Foundation](#1-mdp-foundation)
   - 1.1 [Core Components](#11-core-components)
   - 1.2 [Objective Function](#12-objective-function)
2. [Value Functions](#2-value-functions)
   - 2.1 [State Value Function](#21-state-value-function)
   - 2.2 [Action Value Function](#22-action-value-function)
   - 2.3 [Bellman Equation](#23-bellman-equation)
3. [Policy](#3-policy)
   - 3.1 [Deterministic vs Stochastic](#31-deterministic-vs-stochastic)
   - 3.2 [Policy Gradient](#32-policy-gradient)
4. [RL Paradigms](#4-rl-paradigms)
   - 4.1 [Value-based](#41-value-based)
   - 4.2 [Policy-based](#42-policy-based)
   - 4.3 [Actor-Critic](#43-actor-critic)
5. [Advantage Function](#5-advantage-function)
6. [Exploration vs Exploitation](#6-exploration-vs-exploitation)
7. [Key Algorithms](#7-key-algorithms)
   - 7.1 [DQN](#71-deep-q-network-dqn)
   - 7.2 [PPO](#72-proximal-policy-optimization-ppo)
   - 7.3 [SAC](#73-soft-actor-critic-sac)
   - 7.4 [TD3](#74-twin-delayed-ddpg-td3)
8. [Training Stability](#8-training-stability)
   - 8.1 [Experience Replay](#81-experience-replay)
   - 8.2 [Target Network](#82-target-network)
   - 8.3 [Normalization](#83-normalization)
   - 8.4 [Clipping](#84-clipping)
9. [Reward Engineering](#9-reward-engineering)
   - 9.1 [Sparse vs Dense Reward](#91-sparse-vs-dense-reward)
   - 9.2 [Reward Shaping](#92-reward-shaping)
10. [DeepMimic](#10-deepmimic)
    - 10.1 [Overview](#101-overview)
    - 10.2 [MDP Formulation](#102-mdp-formulation)
    - 10.3 [Imitation Reward](#103-imitation-reward)
    - 10.4 [Reference State Initialization](#104-reference-state-initialization-rsi)
    - 10.5 [Early Termination](#105-early-termination)
    - 10.6 [Network Architecture](#106-network-architecture)
    - 10.7 [Training with PPO](#107-training-with-ppo)

---

## 1. MDP Foundation

### 1.1 Core Components

Reinforcement Learning is formalized as a **Markov Decision Process (MDP)**:

| Symbol | Name | Description |
|--------|------|-------------|
| `s ∈ S` | State | Current observation of the world |
| `a ∈ A` | Action | Decision made by the agent |
| `r` | Reward | Scalar feedback signal |
| `P(s'｜s,a)` | Transition | Probability of next state given current state and action |
| `γ ∈ [0,1)` | Discount | Weight for future rewards; near 1 = long-sighted |

```python
import gymnasium as gym

env = gym.make("CartPole-v1")

obs, info = env.reset()          # s0 — initial state
action = env.action_space.sample()  # a — random action
next_obs, reward, terminated, truncated, info = env.step(action)

print(f"State:   {obs}")
print(f"Action:  {action}")
print(f"Reward:  {reward}")
print(f"Next s:  {next_obs}")
print(f"Done:    {terminated or truncated}")
```

**Markov Property:** The next state depends only on the current state and action — not the full history.

$$P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} \mid s_t, a_t)$$

### 1.2 Objective Function

The agent's goal is to find a policy $\pi$ that maximizes expected discounted cumulative reward:

$$\max_{\pi} \; \mathbb{E}\!\left[\sum_{t=0}^{\infty} \gamma^t \, r_t\right]$$

| Term | Role | Detail |
|------|------|--------|
| $\max_{\pi}$ | **Optimize** | Search over all possible policies and pick the best one |
| $\mathbb{E}[\,\cdot\,]$ | **Expectation** | Average over all randomness — stochastic transitions and stochastic policy |
| $\sum_{t=0}^{\infty}$ | **Cumulative** | Sum rewards over the entire future trajectory, not just the next step |
| $\gamma^t$ | **Discount** | Shrinks the weight of far-future rewards; $\gamma^t \to 0$ as $t \to \infty$, keeping the sum finite |
| $r_t$ | **Reward** | Scalar signal received from the environment at timestep $t$ |

> **Intuition:** find a policy so the agent collects as much reward as possible over time, but rewards received sooner count more than those far in the future.

```python
# Simulating a rollout and computing discounted return
def compute_return(rewards, gamma=0.99):
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

rewards = [1, 1, 1, 10]          # episode rewards
print(compute_return(rewards))   # [12.91, 11.99, 10.99, 10.0]
```

---

## 2. Value Functions

Value functions estimate how good a state (or state-action pair) is under a given policy.

### 2.1 State Value Function

$V^\pi(s)$ — expected return starting from state $s$ and following policy $\pi$:

$$V^{\pi}(s) = \mathbb{E}_{\pi}\!\left[\sum_{t=0}^{\infty} \gamma^t r_t \;\middle|\; s_0 = s\right]$$

| Term | Detail |
|------|--------|
| $V^{\pi}(s)$ | **State value** — how good it is to be in state $s$ under policy $\pi$ |
| $\mathbb{E}_{\pi}[\,\cdot\,]$ | Expectation taken while following $\pi$ — averaging over all trajectories the policy can produce |
| $\sum_{t=0}^{\infty} \gamma^t r_t$ | Discounted cumulative reward from this point onward |
| $\mid s_0 = s$ | Conditioning — we start the sum from state $s$ |

```python
class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)       # outputs a scalar V(s)
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)
```

### 2.2 Action Value Function

$Q^\pi(s,a)$ — expected return starting from state $s$, taking action $a$, then following $\pi$:

$$Q^{\pi}(s,a) = \mathbb{E}_{\pi}\!\left[\sum_{t=0}^{\infty} \gamma^t r_t \;\middle|\; s_0 = s,\; a_0 = a\right]$$

| Term | Detail |
|------|--------|
| $Q^{\pi}(s,a)$ | **Action value** — how good it is to take action $a$ in state $s$, then follow $\pi$ thereafter |
| $\mid s_0 = s,\; a_0 = a$ | First action is fixed to $a$; from $t=1$ onward the policy $\pi$ takes over |
| $Q - V$ relationship | $Q^\pi(s,a) \geq V^\pi(s)$ when $a$ is better than average; equal when $\pi$ is deterministic |

```python
class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)       # outputs a scalar Q(s,a)
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x).squeeze(-1)
```

### 2.3 Bellman Equation

The Bellman equation is the recursive definition at the core of all RL:

$$Q(s,a) = r + \gamma \, \mathbb{E}_{s'}\!\left[\max_{a'} Q(s', a')\right]$$

| Term | Detail |
|------|--------|
| $Q(s,a)$ | Value of the current state-action pair — what we want to learn |
| $r$ | Immediate reward received after taking action $a$ in state $s$ |
| $\gamma$ | Discount factor — scales down the future relative to now |
| $\mathbb{E}_{s'}[\,\cdot\,]$ | Average over all possible next states $s'$ (environment stochasticity) |
| $\max_{a'} Q(s', a')$ | Best possible value from the next state — assumes optimal play from $s'$ onward |

> **Key idea:** the value now equals the immediate reward plus the discounted best value you can get from here. RL solves this equation iteratively.

```python
# TD target — bootstrapped value estimate
def bellman_target(rewards, next_values, dones, gamma=0.99):
    # next_values: V(s') from value network
    # dones: 1 if terminal, 0 otherwise
    return rewards + gamma * next_values * (1 - dones)

# Example
rewards     = torch.tensor([1.0, 1.0, 1.0])
next_values = torch.tensor([5.0, 5.0, 0.0])  # last is terminal
dones       = torch.tensor([0.0, 0.0, 1.0])

targets = bellman_target(rewards, next_values, dones)
# tensor([5.95, 5.95, 1.00])
```

---

## 3. Policy

### 3.1 Deterministic vs Stochastic

```python
# Deterministic policy: directly outputs action
class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, act_dim), nn.Tanh()   # bounded to [-1, 1]
        )

    def forward(self, obs):
        return self.net(obs)

# Stochastic policy: outputs a distribution over actions
class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256),     nn.Tanh()
        )
        self.mean_head = nn.Linear(256, act_dim)
        self.log_std   = nn.Parameter(torch.zeros(act_dim))  # learnable std

    def forward(self, obs):
        h    = self.net(obs)
        mean = self.mean_head(h)
        std  = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        return dist

    def act(self, obs):
        dist   = self.forward(obs)
        action = dist.sample()
        log_p  = dist.log_prob(action).sum(-1)
        return action, log_p
```

| | Deterministic | Stochastic |
|---|---|---|
| Output | Single action | Distribution over actions |
| Exploration | External noise (e.g., ε-greedy) | Built-in via sampling |
| Used by | TD3, DDPG | PPO, SAC |

### 3.2 Policy Gradient

The policy gradient theorem gives the direction to update $\theta$ to increase expected return:

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{\pi}\!\left[\nabla_{\theta} \log \pi_{\theta}(a \mid s) \cdot A(s,a)\right]$$

| Term | Detail |
|------|--------|
| $\nabla_{\theta} J(\theta)$ | Gradient of expected return w.r.t. policy parameters — the direction to step |
| $\nabla_{\theta} \log \pi_{\theta}(a \mid s)$ | **Score function** — how sensitive the log-probability of action $a$ is to $\theta$ |
| $A(s,a)$ | **Advantage** — scales the gradient; positive → push $a$ to be more likely, negative → less likely |
| $\mathbb{E}_{\pi}[\,\cdot\,]$ | Average over trajectories sampled from the current policy |

> **Intuition:** increase the probability of actions that were better than average ($A > 0$), decrease the probability of actions that were worse ($A < 0$).

```python
def policy_gradient_loss(log_probs, advantages):
    # log_probs:   log π_θ(a_t | s_t)  shape: (T,)
    # advantages:  A(s_t, a_t)          shape: (T,)
    return -(log_probs * advantages).mean()   # negative because we maximize

# In the training loop:
dist   = policy(obs)
action = dist.sample()
log_p  = dist.log_prob(action).sum(-1)
loss   = policy_gradient_loss(log_p, advantages)
loss.backward()
optimizer.step()
```

---

## 4. RL Paradigms

### 4.1 Value-based

Learns $Q(s,a)$ and acts greedily: $\pi(s) = \arg\max_{a}\, Q(s,a)$. Works best for **discrete** action spaces.

```python
# ε-greedy action selection
def select_action(q_net, obs, epsilon=0.1):
    if np.random.rand() < epsilon:
        return env.action_space.sample()    # explore
    with torch.no_grad():
        q_values = q_net(obs)
        return q_values.argmax().item()     # exploit
```

### 4.2 Policy-based

Directly optimizes the policy without maintaining a value table. Works for **continuous** action spaces. High variance — requires many samples.

### 4.3 Actor-Critic

Combines both: the **Actor** learns the policy; the **Critic** learns the value function to reduce variance.

```python
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256),     nn.Tanh()
        )
        self.actor_mean = nn.Linear(256, act_dim)
        self.critic     = nn.Linear(256, 1)
        self.log_std    = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        h      = self.shared(obs)
        mean   = self.actor_mean(h)
        value  = self.critic(h).squeeze(-1)
        std    = self.log_std.exp().expand_as(mean)
        dist   = torch.distributions.Normal(mean, std)
        return dist, value
```

| Method | Variance | Bias | Action Space | Examples |
|--------|----------|------|--------------|---------|
| Value-based | Low | High | Discrete | DQN |
| Policy-based | High | Low | Continuous | REINFORCE |
| Actor-Critic | Low | Low | Both | PPO, SAC, TD3 |

---

## 5. Advantage Function

The advantage $A(s,a)$ measures how much better action $a$ is compared to the average action in state $s$:

$$A(s,a) = Q(s,a) - V(s)$$

| Term | Detail |
|------|--------|
| $Q(s,a)$ | Value of taking this specific action $a$ |
| $V(s)$ | Baseline — average value of state $s$ regardless of action |
| $A > 0$ | Action $a$ was better than the average action in $s$ |
| $A < 0$ | Action $a$ was worse than the average action in $s$ |

Using raw returns leads to high variance. **Generalized Advantage Estimation (GAE)** smooths this:

$$A_t^{\,\text{GAE}} = \sum_{k=0}^{\infty} (\gamma\lambda)^k \,\delta_{t+k}$$

$$\delta_t = r_t + \gamma\,V(s_{t+1}) - V(s_t) \qquad (\text{TD error})$$

| Term | Detail |
|------|--------|
| $\delta_t$ | **TD error** — one-step advantage signal; how much better $s_{t+1}$ turned out to be than predicted |
| $(\gamma\lambda)^k$ | Exponential decay — nearby TD errors weighted more heavily than distant ones |
| $\lambda \in [0,1]$ | **Trace decay**: $\lambda=0$ gives pure TD (low variance, high bias); $\lambda=1$ gives full return (high variance, low bias) |

```python
def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0.0
    values = values + [next_value]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae   = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns
```

**Key insight:** $A(s,a) > 0$ → action was better than average, increase its probability. $A(s,a) < 0$ → worse than average, decrease its probability.

---

## 6. Exploration vs Exploitation

| Strategy | Mechanism | Use Case |
|----------|-----------|---------|
| ε-greedy | Random action with probability ε | DQN, discrete |
| Entropy bonus | Maximize policy entropy | SAC, continuous |
| Noise injection | Add Gaussian noise to action | TD3, DDPG |
| UCB | Prefer high-uncertainty states | Bandit, model-based |

```python
# SAC entropy bonus — encourages diverse action selection
def sac_actor_loss(policy, q_net1, q_net2, obs, alpha=0.2):
    dist   = policy(obs)
    action = dist.rsample()                          # reparameterization trick
    log_p  = dist.log_prob(action).sum(-1)

    q1 = q_net1(obs, action)
    q2 = q_net2(obs, action)
    q  = torch.min(q1, q2)

    return (alpha * log_p - q).mean()               # maximize Q, maximize entropy
```

---

## 7. Key Algorithms

### 7.1 Deep Q-Network (DQN)

Uses a neural network to approximate `Q(s,a)`. Designed for **discrete** action spaces.

```python
class DQN(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),     nn.ReLU(),
            nn.Linear(128, n_actions)            # one Q-value per action
        )

    def forward(self, obs):
        return self.net(obs)

def dqn_loss(q_net, target_net, batch, gamma=0.99):
    obs, actions, rewards, next_obs, dones = batch

    q_values      = q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q    = target_net(next_obs).max(1).values
        td_target = rewards + gamma * next_q * (1 - dones)

    return nn.functional.mse_loss(q_values, td_target)
```

### 7.2 Proximal Policy Optimization (PPO)

The industry standard for stable, sample-efficient RL. Clips the policy update ratio to prevent destructive updates.

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}\!\left[\min\!\left(r_t(\theta)\,A_t,\;\operatorname{clip}\!\left(r_t(\theta),\,1{-}\varepsilon,\,1{+}\varepsilon\right)A_t\right)\right]$$

$$r_t(\theta) = \frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

| Term | Detail |
|------|--------|
| $r_t(\theta)$ | **Probability ratio** — how much more (or less) likely the new policy is to take the same action |
| $r_t = 1$ | New and old policy agree exactly |
| $r_t > 1$ | New policy made this action more likely |
| $r_t < 1$ | New policy made this action less likely |
| $A_t$ | Advantage — determines whether we want to push the action up or down |
| $\operatorname{clip}(\cdot,\, 1{-}\varepsilon,\, 1{+}\varepsilon)$ | Hard constraint — ratio is forced to stay within $[1-\varepsilon,\, 1+\varepsilon]$ (typically $\varepsilon = 0.2$) |
| $\min(\cdot,\cdot)$ | Take the pessimistic bound — prevents the update from being too greedy in either direction |

> **Key idea:** PPO lets the policy improve, but won't allow a single update to change it so drastically that it becomes unstable.

```python
def ppo_loss(new_log_probs, old_log_probs, advantages, clip_eps=0.2):
    ratio   = (new_log_probs - old_log_probs).exp()   # π_new / π_old

    surr1   = ratio * advantages
    surr2   = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages

    return -torch.min(surr1, surr2).mean()            # negative: we maximize

def ppo_value_loss(values, returns):
    return nn.functional.mse_loss(values, returns)

def ppo_entropy_bonus(dist):
    return dist.entropy().sum(-1).mean()

# Total PPO loss
def total_ppo_loss(policy_loss, value_loss, entropy, c1=0.5, c2=0.01):
    return policy_loss + c1 * value_loss - c2 * entropy
```

### 7.3 Soft Actor-Critic (SAC)

Off-policy Actor-Critic for **continuous** control. Maximizes reward + entropy (exploration built in). Very sample-efficient.

$$J(\pi) = \mathbb{E}\!\left[\sum_{t} \gamma^t \Bigl(r_t + \alpha\,\mathcal{H}\bigl(\pi(\cdot \mid s_t)\bigr)\Bigr)\right]$$

| Term | Detail |
|------|--------|
| $r_t$ | Environment reward at step $t$ |
| $\mathcal{H}(\pi(\cdot \mid s_t))$ | **Entropy** of the policy at $s_t$ — high entropy means the policy spreads probability across many actions |
| $\alpha$ | **Temperature** — controls the trade-off between reward and exploration; higher $\alpha$ = more random |
| $r_t + \alpha\,\mathcal{H}$ | SAC maximizes reward AND entropy simultaneously, so exploration is built into the objective |

```python
# SAC uses two Q-networks to reduce overestimation bias
class SAC:
    def __init__(self, obs_dim, act_dim, alpha=0.2):
        self.actor   = GaussianPolicy(obs_dim, act_dim)
        self.q1      = QNet(obs_dim, act_dim)
        self.q2      = QNet(obs_dim, act_dim)
        self.q1_targ = QNet(obs_dim, act_dim)
        self.q2_targ = QNet(obs_dim, act_dim)
        self.alpha   = alpha             # entropy temperature

    def critic_loss(self, obs, actions, rewards, next_obs, dones, gamma=0.99):
        with torch.no_grad():
            dist         = self.actor(next_obs)
            next_action  = dist.rsample()
            next_log_p   = dist.log_prob(next_action).sum(-1)

            q1_next = self.q1_targ(next_obs, next_action)
            q2_next = self.q2_targ(next_obs, next_action)
            q_next  = torch.min(q1_next, q2_next) - self.alpha * next_log_p
            target  = rewards + gamma * (1 - dones) * q_next

        loss1 = nn.functional.mse_loss(self.q1(obs, actions), target)
        loss2 = nn.functional.mse_loss(self.q2(obs, actions), target)
        return loss1 + loss2
```

### 7.4 Twin Delayed DDPG (TD3)

Deterministic policy for **continuous** control. Fixes overestimation bias (twin Q-nets) and instability (delayed policy updates).

```python
# TD3 key ideas:
# 1. Twin critics — take min to reduce overestimation
# 2. Delayed actor update — update actor every 2 critic steps
# 3. Target policy smoothing — add noise to target actions

def td3_critic_loss(q1, q2, actor_targ, q1_targ, q2_targ,
                    obs, actions, rewards, next_obs, dones,
                    gamma=0.99, noise_std=0.2, noise_clip=0.5):
    with torch.no_grad():
        noise       = torch.randn_like(actions) * noise_std
        noise       = noise.clamp(-noise_clip, noise_clip)
        next_action = (actor_targ(next_obs) + noise).clamp(-1, 1)

        q1_next = q1_targ(next_obs, next_action)
        q2_next = q2_targ(next_obs, next_action)
        target  = rewards + gamma * (1 - dones) * torch.min(q1_next, q2_next)

    loss = (nn.functional.mse_loss(q1(obs, actions), target) +
            nn.functional.mse_loss(q2(obs, actions), target))
    return loss
```

| Algorithm | Action Space | On/Off-policy | Key Feature |
|-----------|-------------|---------------|-------------|
| DQN | Discrete | Off | Experience replay + target net |
| PPO | Both | On | Clipped policy update |
| SAC | Continuous | Off | Entropy maximization |
| TD3 | Continuous | Off | Twin critics + delayed update |

---

## 8. Training Stability

### 8.1 Experience Replay

Stores past transitions and samples mini-batches to break temporal correlation.

```python
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (torch.FloatTensor(obs),
                torch.FloatTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(next_obs),
                torch.FloatTensor(dones))

    def __len__(self):
        return len(self.buffer)
```

### 8.2 Target Network

A slowly updated copy of the Q-network used to compute stable TD targets. Prevents the "moving target" problem.

```python
# Hard update — copy weights every N steps
def hard_update(target_net, source_net):
    target_net.load_state_dict(source_net.state_dict())

# Soft update — exponential moving average (τ ≈ 0.005)
def soft_update(target_net, source_net, tau=0.005):
    for t_param, s_param in zip(target_net.parameters(), source_net.parameters()):
        t_param.data.copy_(tau * s_param.data + (1 - tau) * t_param.data)

# Call after each critic update:
soft_update(q1_targ, q1)
soft_update(q2_targ, q2)
```

### 8.3 Normalization

```python
class RunningNorm:
    def __init__(self, shape, eps=1e-8):
        self.mean  = np.zeros(shape)
        self.var   = np.ones(shape)
        self.count = eps

    def update(self, x):
        batch_mean = x.mean(axis=0)
        batch_var  = x.var(axis=0)
        batch_n    = x.shape[0]

        total_n    = self.count + batch_n
        delta      = batch_mean - self.mean
        self.mean += delta * batch_n / total_n
        self.var   = (self.var * self.count + batch_var * batch_n +
                      delta**2 * self.count * batch_n / total_n) / total_n
        self.count = total_n

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

# Normalize observations and rewards separately
obs_norm    = RunningNorm(obs_dim)
reward_norm = RunningNorm(1)
```

### 8.4 Clipping

```python
# PPO ratio clipping — prevents too-large policy updates
ratio_clipped = ratio.clamp(1 - 0.2, 1 + 0.2)

# Gradient clipping — prevents exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

# Reward clipping — stabilizes training on diverse reward scales
reward = np.clip(reward, -10.0, 10.0)
```

---

## 9. Reward Engineering

### 9.1 Sparse vs Dense Reward

```python
# Sparse: only signals success/failure
def sparse_reward(reached_goal):
    return 1.0 if reached_goal else 0.0

# Dense: provides continuous guidance signal
def dense_reward(agent_pos, goal_pos, alive):
    if not alive:
        return -100.0
    distance = np.linalg.norm(agent_pos - goal_pos)
    return -distance                # closer = better
```

| | Sparse | Dense |
|---|---|---|
| Difficulty | Hard to learn (delayed signal) | Easier to learn |
| Risk | Requires long exploration | Can cause reward hacking |
| Use case | Precise tasks | Continuous control |

### 9.2 Reward Shaping

Adds potential-based bonuses to guide learning without changing the optimal policy.

$$r'(s,a,s') = r(s,a,s') + F(s,s'), \qquad F(s, s') = \gamma\,\Phi(s') - \Phi(s)$$

| Term | Detail |
|------|--------|
| $\Phi(s)$ | **Potential function** — a hand-designed score for state $s$ (e.g., negative distance to goal) |
| $\gamma\,\Phi(s')$ | Potential of the next state, discounted |
| $F(s, s')$ | Shaping bonus — positive when moving to a better state, negative when moving to a worse one |
| $r'$ | Shaped reward — guaranteed to have the same optimal policy as the original $r$ |

```python
def shaped_reward(r, phi_s, phi_next_s, gamma=0.99):
    # phi(s): potential function (e.g., negative distance to goal)
    # F(s, s') = γ·φ(s') - φ(s)  — guaranteed not to change optimal policy
    shaping = gamma * phi_next_s - phi_s
    return r + shaping

# Example: guide agent toward a target position
phi = lambda pos, goal: -np.linalg.norm(pos - goal)
r_shaped = shaped_reward(r_env, phi(pos, goal), phi(next_pos, goal))
```

---

## 10. DeepMimic

### 10.1 Overview

**DeepMimic** (Peng et al., 2018) is a physics-based character animation system that trains a simulated character to reproduce motion capture clips using deep RL. The character learns to mimic reference poses while remaining physically plausible under a physics simulator (e.g., PyBullet, MuJoCo).

Key contributions:
- **Imitation reward** — directly compares simulated pose to reference motion
- **Reference State Initialization (RSI)** — initializes each episode at a random point in the reference clip
- **Early termination** — ends episodes when the character falls, accelerating learning

### 10.2 MDP Formulation

```
State s:      joint positions, joint velocities, end-effector positions, phase φ ∈ [0,1]
Action a:     target joint angles (PD controller drives joints toward targets)
Reward r:     weighted imitation reward (pose, velocity, end-effector, root)
Transition:   physics simulator (PyBullet / MuJoCo)
```

```python
# Observation: concatenate kinematic features + phase
def build_observation(sim_char, phase):
    joint_pos   = sim_char.get_joint_positions()    # (J,) joint angles
    joint_vel   = sim_char.get_joint_velocities()   # (J,) angular velocities
    ee_pos      = sim_char.get_end_effector_positions()  # (4, 3) wrist/ankle positions
    root_pos    = sim_char.get_root_position()      # (3,)
    root_vel    = sim_char.get_root_velocity()      # (6,) linear + angular

    return np.concatenate([joint_pos, joint_vel, ee_pos.flatten(),
                           root_pos, root_vel, [phase]])

# Action: PD controller target angles
def apply_action(sim_char, action, kp=300, kd=20):
    target_angles = action                           # network output
    sim_char.set_pd_targets(target_angles, kp, kd)
```

### 10.3 Imitation Reward

The total reward is a weighted sum of four sub-rewards:

$$r_t = w_p \cdot r^p + w_v \cdot r^v + w_e \cdot r^e + w_c \cdot r^c$$

| Term | Detail |
|------|--------|
| $w_p, w_v, w_e, w_c$ | Scalar weights that control the relative importance of each sub-reward |
| $r^p,\, r^v,\, r^e,\, r^c$ | Sub-rewards in $[0, 1]$ — each equals 1 when perfectly matched, decays toward 0 on deviation |

Each sub-reward uses a Gaussian kernel so it equals 1 when perfectly matched and decays toward 0 on deviation:

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

| Term | Weight | Measures |
|------|--------|---------|
| $r^p$ | 0.65 | Joint angle similarity |
| $r^v$ | 0.10 | Joint velocity similarity |
| $r^e$ | 0.15 | Wrist/ankle position similarity |
| $r^c$ | 0.10 | Pelvis position/orientation |

```python
def imitation_reward(sim, ref, weights=(0.65, 0.1, 0.15, 0.1)):
    w_p, w_v, w_e, w_c = weights

    dq  = sim['joint_pos'] - ref['joint_pos']
    r_p = np.exp(-2.0  * np.sum(dq ** 2))

    dv  = sim['joint_vel'] - ref['joint_vel']
    r_v = np.exp(-0.1  * np.sum(dv ** 2))

    de  = sim['ee_pos'] - ref['ee_pos']
    r_e = np.exp(-40.0 * np.sum(de ** 2))

    dc  = sim['root_pos'] - ref['root_pos']
    r_c = np.exp(-10.0 * np.sum(dc ** 2))

    return w_p * r_p + w_v * r_v + w_e * r_e + w_c * r_c
```

### 10.4 Reference State Initialization (RSI)

Instead of always starting at the beginning of the clip, each episode starts at a random phase $\varphi \sim \mathcal{U}[0,1)$. This exposes the agent to all parts of the motion early in training.

```python
import random

class MotionClip:
    def __init__(self, frames, fps=30):
        self.frames = frames          # list of reference poses
        self.fps    = fps

    def sample_init_state(self):
        phase = random.random()       # uniform [0, 1)
        frame_idx = int(phase * len(self.frames))
        return self.frames[frame_idx], phase

    def get_reference_at(self, phase):
        idx = int(phase * len(self.frames)) % len(self.frames)
        return self.frames[idx]

    def advance_phase(self, phase, dt):
        duration = len(self.frames) / self.fps
        return (phase + dt / duration) % 1.0

# Episode reset with RSI
def reset(sim_char, clip):
    ref_pose, phase = clip.sample_init_state()
    sim_char.set_state(ref_pose)       # initialize sim to reference pose
    return build_observation(sim_char, phase), phase
```

### 10.5 Early Termination

The episode ends if the character falls (root height below threshold), preventing the agent from wasting time on failed states.

```python
def check_early_termination(sim_char, min_root_height=0.5):
    root_height = sim_char.get_root_position()[2]   # z-axis height
    return root_height < min_root_height

# In the environment step:
def step(sim_char, clip, action, phase, dt=1/30):
    sim_char.apply_action(action)
    sim_char.simulate(dt)

    next_phase = clip.advance_phase(phase, dt)
    ref        = clip.get_reference_at(next_phase)

    obs    = build_observation(sim_char, next_phase)
    reward = imitation_reward(sim_char.get_state(), ref)
    done   = check_early_termination(sim_char)

    return obs, reward, done, next_phase
```

### 10.6 Network Architecture

Both actor and critic are fully connected networks with joint inputs. Tanh activations are typical for physics-based control.

```python
class DeepMimicActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512),     nn.ReLU(),
        )
        self.mean_head = nn.Linear(512, act_dim)
        self.log_std   = nn.Parameter(torch.full((act_dim,), -2.0))  # small initial std

    def forward(self, obs):
        h    = self.net(obs)
        mean = self.mean_head(h)
        std  = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

class DeepMimicCritic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512),     nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)
```

### 10.7 Training with PPO

DeepMimic uses PPO as the base RL algorithm.

```python
def train_deepmimic(env, actor, critic, clip, n_steps=4096, n_epochs=10,
                    lr=3e-4, gamma=0.99, lam=0.95, clip_eps=0.2):

    actor_opt  = torch.optim.Adam(actor.parameters(),  lr=lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)

    # Collect rollout
    obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

    obs, phase = env.reset()
    for _ in range(n_steps):
        obs_t  = torch.FloatTensor(obs)
        dist   = actor(obs_t)
        action = dist.sample()
        logp   = dist.log_prob(action).sum(-1)
        value  = critic(obs_t)

        next_obs, reward, done, phase = env.step(action.numpy(), phase)

        obs_buf.append(obs_t);      act_buf.append(action)
        logp_buf.append(logp);      rew_buf.append(reward)
        val_buf.append(value);      done_buf.append(float(done))

        obs = next_obs
        if done:
            obs, phase = env.reset()

    # Compute GAE
    next_val   = critic(torch.FloatTensor(obs)).detach()
    advantages, returns = compute_gae(rew_buf, [v.item() for v in val_buf],
                                      next_val.item(), done_buf, gamma, lam)

    obs_t   = torch.stack(obs_buf)
    act_t   = torch.stack(act_buf)
    old_lp  = torch.stack(logp_buf).detach()
    adv_t   = torch.FloatTensor(advantages)
    ret_t   = torch.FloatTensor(returns)
    adv_t   = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)   # normalize

    # PPO update
    for _ in range(n_epochs):
        dist     = actor(obs_t)
        new_lp   = dist.log_prob(act_t).sum(-1)
        values   = critic(obs_t)

        p_loss   = ppo_loss(new_lp, old_lp, adv_t, clip_eps)
        v_loss   = nn.functional.mse_loss(values, ret_t)
        entropy  = dist.entropy().sum(-1).mean()

        actor_opt.zero_grad()
        (p_loss - 0.01 * entropy).backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        actor_opt.step()

        critic_opt.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
        critic_opt.step()

    return p_loss.item(), v_loss.item()
```

**Key hyperparameters for DeepMimic:**

| Parameter | Value | Note |
|-----------|-------|------|
| `gamma` | 0.95 | Lower than typical — motion is short-horizon |
| `lam` (GAE) | 0.95 | Standard |
| `clip_eps` | 0.2 | Standard PPO |
| `n_steps` | 4096+ | More steps per update for stable physics |
| `n_epochs` | 10 | Reuse each rollout 10 times |
| `lr` | 5e-5 | Small for fine physical control |
| reward weights | (0.65, 0.1, 0.15, 0.1) | Pose-heavy |
