# SuperDog

**SuperDog** is a path-planning repository for the **Unitree A1** quadruped robot. A high-level **Soft Actor-Critic (SAC)** policy generates velocity commands, and a frozen pretrained walking policy converts them into 12-DOF joint torques. The robot navigates from a random spawn point to a goal in a room with randomly placed obstacles.

> **Where to start**
>
> - Run & install instructions: [RUNME.md](RUNME.md)
> - Robot config and SAC hyperparameters: [configs/a1.yaml](configs/a1.yaml)
> - Curriculum learning schedule: [configs/curriculum.yaml](configs/curriculum.yaml)

<p align="center">
  <img src="docs/photo_2026-03-26_04-57-41.jpg" alt="SuperDog simulation 1" width="45%" />
  <img src="docs/photo_2026-03-26_04-57-54.jpg" alt="SuperDog simulation 2" width="45%" />
</p>

<a id="table-of-contents"></a>

## Table of Contents

1. [Overview](#overview)
2. [Environment](#environment)
3. [Architecture](#architecture)
4. [Observation Space](#observation-space)
5. [Action Space](#action-space)
6. [Reward Function](#reward-function)
7. [Curriculum Learning](#curriculum-learning)
8. [Hyperparameters](#hyperparameters)

---

<a id="overview"></a>

## Overview

SuperDog trains a navigation policy through **curriculum learning** across 7 difficulty levels, progressively increasing obstacle count and tightening reward shaping.

### Policy stack

| Layer | Role | Frequency |
|---|---|---|
| SAC Actor | Generates `[vx, vy, ω]` velocity command | 10 Hz (configurable) |
| Walking Policy | Converts command to 12 joint torques | 50 Hz |
| MuJoCo Simulator | Physics step | 500 Hz |

### Core definitions used throughout the README

| Term | Definition |
|---|---|
| **Episode** | One complete rollout from spawn to goal-reached, collision, or timeout. |
| **Success rate** | Fraction of episodes where the robot reaches the goal. Averaged over the last 100 episodes. |
| **Curriculum level** | A difficulty stage that sets obstacle count, episode length, reward weights, and SAC hyperparameters. Advances when `success_rate` exceeds the level threshold. |
| **Asymmetric actor-critic** | The actor sees only sensor data available on a real robot (with noise). The critic sees clean privileged data with additional features (velocity, nearest obstacles). |

---

<a id="environment"></a>

## Environment

<p align="center">
  <img src="docs/photo_2026-03-26_04-57-41.jpg" alt="Environment with obstacles" width="45%" />
  <img src="docs/photo_2026-03-26_04-57-54.jpg" alt="Robot navigating" width="45%" />
</p>

| Property | Value |
|---|---|
| Robot | Unitree A1 (12 DOF) |
| Simulator | MuJoCo 3.2.3 |
| Room size | Fixed rectangular room |
| Obstacles | 3–20 random boxes (depends on curriculum level) |
| Episode length | 6 000–17 000 steps (curriculum-dependent) |
| Spawn | Random free-space point with clearance ≥ 0.7 m |
| Goal | Random free-space point |

---

<a id="architecture"></a>

## Architecture

The architecture uses **asymmetric actor-critic**: the critic receives additional privileged information (velocity, obstacle proximity) that the actor does not see — enabling better value estimation while the actor remains deployable on a real robot with limited sensors.

```
Observation (lidar sectors + goal vector + proprioception)
        │
        ▼
┌──────────────┐     action history (last 3 steps)
│  SAC Actor   │◄────────────────────────────────┐
│ [512,512,256]│                                 │
└──────┬───────┘                                 │
       │ cmd = [vx, vy, ω]                       │
       ▼                                         │
┌──────────────┐                                 │
│Walking Policy│  (frozen, pretrained)            │
│   12-DOF PD  │                                 │
└──────┬───────┘                                 │
       │ joint torques                           │
       ▼                                         │
┌──────────────┐                                 │
│   MuJoCo     │──► next observation ────────────┘
│  Simulator   │
└──────────────┘
```

### Network Architecture

The full architecture diagram is available as an interactive HTML page: [assets/architecture.html](assets/architecture.html).

| Network | Input | Hidden layers | Output |
|---|---|---|---|
| Actor | 56 dims | 512 → 512 → 256 (ReLU) | 6 → split into μ(3), log_σ(3) → SquashedNormal |
| Critic Q₁ | 204 dims (obs + action) | 512 → 512 → 256 (ReLU) | 1 (Q-value) |
| Critic Q₂ | 204 dims (obs + action) | 512 → 512 → 256 (ReLU) | 1 (Q-value) |

The critic uses **clipped double-Q**: `min(Q₁, Q₂)` is used for the actor update.

---

<a id="observation-space"></a>

## Observation Space

The project uses **asymmetric observations**: the actor sees only sensor data available on a real robot (with noise), while the critic sees clean privileged data with additional features.

### Actor observation (47 → 56 dims)

| # | Component | Dim | Normalization |
|---|---|---|---|
| 1 | LiDAR sectors | 40 | `[0, max_range]` → `[-1, 1]` |
| 2 | Angular velocity ω | 1 | `/ max_angular_vel` |
| 3 | sin(angle to target) | 1 | clipped `[-1, 1]` |
| 4 | cos(angle to target) | 1 | clipped `[-1, 1]` |
| 5 | Distance to target | 1 | `[0, max_dist]` → `[-1, 1]` |
| 6 | Previous action `[vx, vy, ω]` | 3 | clipped `[-1, 1]` |
| | **Base total** | **47** | |
| 7 | Action history (last 3 steps × 3) | 9 | appended to base |
| | **Actor network input** | **56** | |

During training, **observation noise** is injected for sim-to-real transfer:

| Signal | Noise σ |
|---|---|
| Distance to target | 0.05 m |
| Angle to target | 0.05 rad (~3°) |
| Angular velocity | 0.02 rad/s |
| LiDAR | 0.02 m |

### Critic observation (49 → 201 dims)

The critic builds on the actor observation but with **no noise** (clean privileged data) and additional features:

| # | Component | Dim | Notes |
|---|---|---|---|
| 1 | Actor base observation (clean) | 47 | Same layout, zero noise |
| 2 | Forward velocity vx | 1 | **Privileged** — not available to actor |
| 3 | Lateral velocity vy | 1 | **Privileged** — not available to actor |
| | **Critic base (single frame)** | **49** | |
| 4 | History stacking: 4 frames (t, t−1, t−2, t−3) | 196 | 49 × 4 |
| 5 | Top-5 nearest LiDAR distances | 5 | **Privileged**: sorted raw beams, normalized |
| | **Critic network input** | **201** | + action(3) = 204 into Q-network |

```
Actor  input = base_obs(47) + action_history(3 × 3)                     = 56  dims
Critic input = base_obs(49) × 4 frames + top_k(5) + action(3)           = 204 dims
               └────────── 196 ──────────┘  └─5─┘    └─3─┘
```

The **top-5 features** are the 5 smallest raw LiDAR distances (normalized to `[-1, 1]`), giving the critic explicit awareness of the nearest obstacles for better value estimation.

---

<a id="action-space"></a>

## Action Space

The SAC actor outputs a 3-dimensional continuous command, squashed through `tanh` into `[-1, 1]`:

| Dim | Variable | Scale |
|---|---|---|
| 0 | `vx` — forward velocity | `cmd_scale[0] = 1.0` |
| 1 | `vy` — lateral velocity | `cmd_scale[1] = 0.5` |
| 2 | `ω` — yaw rate | `cmd_scale[2] = 0.35` |

---

<a id="reward-function"></a>

## Reward Function

| Term | Weight | Description |
|---|---|---|
| `reached` | +100.0 | Terminal: goal reached (dist < 0.2 m) |
| `collision` | −500.0 | Terminal: collision (dist to obstacle < 0.35 m) |
| `progress` | +20.0 | Distance reduction toward goal per step |
| `obs_penalty` | −5.0 × f(d) | Exponential penalty for proximity to obstacles |
| `time_penalty` | −0.02 | Per-step penalty to discourage slow trajectories |
| `vx_backward` | −1.0 | Penalty for negative forward velocity |
| `vy_penalty` | −1.0 | Penalty for excessive lateral velocity |
| `velocity_alignment` | +0.5 | Reward for moving in the facing direction |

The obstacle proximity penalty uses an exponential envelope:

```
obs_penalty = obs_penalty_weight × exp(obstacle_exponential_scale × (1 − d / obstacle_threshold))
```

where `d` is the distance to the nearest obstacle and the penalty activates when `d < obstacle_threshold = 1.5 m`.

Reward weights are overridden per curriculum level (see [configs/curriculum.yaml](configs/curriculum.yaml)).

---

<a id="curriculum-learning"></a>

## Curriculum Learning

Training progresses through 7 levels. A level advances when `success_rate` (averaged over the last 100 episodes) exceeds the level threshold. Each level independently adjusts obstacle count, episode length, reward weights, and SAC hyperparameters.

| Level | Name | Obstacles | Max Steps | Success Target |
|---|---|---|---|---|
| 1 | Initial Learning | 3–5 | 6 000 | 20% |
| 2 | Obstacle Awareness | 4–7 | 7 500 | 40% |
| 3 | Collision Reduction | 6–9 | 9 000 | 60% |
| 4 | Efficiency Optimization | 8–12 | 11 000 | 88% |
| 5 | Mastery | 10–14 | 13 000 | 95% |
| 6 | Perfection | 12–17 | 15 000 | 95% |
| 7 | Polish & Perfection | 15–20 | 17 000 | 97% |

<details>
<summary><b>Click to expand per-level reward and SAC overrides</b></summary>

| Parameter | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---|---|---|---|---|---|---|
| **Reward weights** | | | | | | | |
| collision | −300 | −400 | −500 | −600 | −800 | −1 000 | −1 500 |
| obs_penalty_weight | 3.0 | 4.0 | 5.0 | 5.5 | 6.0 | 7.0 | 8.0 |
| obstacle_threshold | 1.6 | 1.5 | 1.5 | 1.4 | 1.3 | 1.2 | 1.15 |
| progress | 25 | 22 | 20 | 25 | 30 | 35 | 40 |
| time_penalty | −0.01 | −0.015 | −0.02 | −0.1 | −0.1 | −0.15 | −0.16 |
| vx_backward_penalty | — | — | −2.5 | −3.0 | −3.0 | −5.0 | −6.0 |
| velocity_alignment | — | — | 0.7 | — | — | — | 2.0 |
| **SAC overrides** | | | | | | | |
| actor_lr | 2e-4 | 1.5e-4 | 1e-4 | 8e-5 | 5e-5 | 3e-5 | 1e-5 |
| critic_lr | 5e-4 | 4e-4 | 3e-4 | 2.5e-4 | 2e-4 | 1.5e-4 | 1e-4 |
| init_temperature | 0.1 | 0.15 | 0.2 | 0.3 | 0.4 | 0.6 | 0.8 |
| batch_size | 128 | 192 | 256 | 320 | 384 | 448 | 512 |
| training_iterations | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
| **Replay buffer** | | | | | | | |
| success_weight | 1.5 | 2.0 | 1.5 | 1.5 | 1.0 | 1.5 | 2.0 |
| collision_weight | 1.5 | 1.5 | 2.0 | 3.0 | 5.0 | 10.0 | 15.0 |

Full configuration: [configs/curriculum.yaml](configs/curriculum.yaml).

</details>

---

<a id="hyperparameters"></a>

## Hyperparameters

<details>
<summary><b>Click to expand SAC hyperparameters table</b></summary>

| Hyperparameter | Value |
|---|---|
| **Network** | |
| Actor hidden dims | [512, 512, 256] |
| Critic hidden dims | [512, 512, 256] |
| Weight init | Orthogonal |
| Actor log_std bounds | [−2, 2] |
| **Optimization** | |
| Actor learning rate | 1e-4 |
| Critic learning rate | 3e-4 |
| Alpha learning rate | 3e-4 |
| All betas | (0.9, 0.999) |
| Discount γ | 0.99 |
| Critic soft update τ | 0.005 |
| Actor update frequency | every step |
| Critic target update frequency | every 2 steps |
| **Entropy** | |
| Initial temperature α | 0.2 |
| Target entropy | −1.5 |
| Learnable temperature | yes |
| **Replay buffer** | |
| Buffer size | 1 000 000 |
| Batch size | 512 |
| Training iterations per step | 4 |
| Min buffer size before training | 5 000 |
| **History** | |
| Actor history (action) | 3 steps |
| Critic history (observation) | 3 frames |
| Critic top-k nearest | 5 |

These are the base hyperparameters from [configs/a1.yaml](configs/a1.yaml). Each curriculum level overrides a subset of them — see [configs/curriculum.yaml](configs/curriculum.yaml) for per-level values.

</details>
