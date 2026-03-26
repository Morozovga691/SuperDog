# 🧭 GreedyMiniGrid

**GreedyMiniGrid** is a presentation-first reinforcement learning repository about solving several **MiniGrid** tasks with two different algorithms:

- **Value Iteration**, a model-based planning method, on `MiniGrid-DoorKey-8x8-v0`
- **ε-Greedy TD Q-Learning**, a tabular control method, on `MiniGrid-Unlock-v0`, `MiniGrid-DoorKey-8x8-v0`, and `MiniGrid-Dynamic-Obstacles-8x8-v0`

In this repository, the word **game** means a concrete **MiniGrid environment**, that is, one fixed task definition with its own map rules, observations, actions, and success condition.

> 🔗 **Where to start**
>
> - Run instructions: [RUNME.md](RUNME.md)
> - Prompt log for how the repository was built: [PROMPTS.md](PROMPTS.md)
> - Generated artifact index: [doc/report.md](doc/report.md)
> - Official MiniGrid website: [minigrid.farama.org](https://minigrid.farama.org/)

<p align="center">
  <img src="doc/q_learning/unlock/success_01.gif" alt="Unlock Q-learning success rollout" width="45%" />
  <img src="doc/value_iteration/doorkey/success_01.gif" alt="DoorKey value-iteration success rollout" width="45%" />
</p>

<a id="table-of-contents"></a>

## 📑 Table of Contents

1. [🎯 Overview](#overview)
2. [🎮 Environments Used in GreedyMiniGrid](#environments)
3. [🧰 Main Dependencies and Project Entry Points](#dependencies)
4. [📐 Mathematics and Notation](#mathematics)
5. [🧠 Value Iteration](#value-iteration)
6. [📚 ε-Greedy TD Q-Learning](#epsilon-greedy-td-q-learning)
7. [🖼️ Artifact Gallery](#artifact-gallery)
8. [⚙️ Commands](#commands)
9. [💬 Prompt Log](#prompt-log)
10. [📚 References](#references)

<a id="overview"></a>

## 🎯 Overview

GreedyMiniGrid is a reproducible lab for comparing **planning** and **learning from experience** on compact grid-world tasks. The code lives in a clean `uv`-managed subproject, the training and evaluation pipelines are script-based, and the repository keeps both **machine-readable artifacts** and **human-readable visual reports**.

### What the repository currently contains

| Track | Environment | Algorithm | Main visual outputs |
| --- | --- | --- | --- |
| Planner | `MiniGrid-DoorKey-8x8-v0` | Value Iteration | 1 canonical success GIF and 5 analysis figures |
| Learner | `MiniGrid-Unlock-v0` | ε-Greedy TD Q-Learning | 2 success GIFs, 5 analysis plots |
| Learner | `MiniGrid-DoorKey-8x8-v0` | ε-Greedy TD Q-Learning | 2 success GIFs, 5 analysis plots |
| Learner | `MiniGrid-Dynamic-Obstacles-8x8-v0` | ε-Greedy TD Q-Learning | 2 success GIFs, 5 analysis plots |

### Core definitions used throughout the README

The repository uses a few metrics many times. Every term is defined here before it is reused later.

| Term | Definition |
| --- | --- |
| **State** | A state is the information used by the algorithm to decide what to do next. In this project, the planner uses an explicit tuple `(x, y, d, has_key, door_open)`, while Q-learning uses an encoded observation flattened into a hashable tuple plus direction. |
| **Action** | An action is one discrete command sent to the environment, for example `left`, `right`, `forward`, `pickup`, or `toggle`. |
| **Reward** | A reward is the scalar feedback obtained after one action. In evaluation plots, reward means the raw environment reward. In Q-learning training plots, reward means the shaped reward used for learning. |
| **Return** | A return is the sum of rewards collected over an episode, usually with discounting in the theoretical definition. In the plots here, episode reward means the accumulated reward for one episode. |
| **Episode** | An episode is one complete rollout from `env.reset()` until termination or truncation. |
| **Discount factor** | The discount factor, written as `γ` (gamma), controls how strongly future reward contributes to the current value estimate. |
| **Learning rate** | The learning rate, written as `α` (alpha), controls how strongly a new sample changes the old Q-value. |
| **Success rate** | Success rate is the fraction of evaluation episodes that achieve a positive raw environment reward. In this repository, evaluation uses 100 episodes by default. |
| **Q-table size** | Q-table size is the number of unique encoded states currently stored as keys in the Python dictionary used by Q-learning. |
| **Evaluation reward distribution** | Evaluation reward distribution is the density plot of raw environment rewards collected across evaluation episodes. Its x-axis is raw environment reward, and its y-axis is density. |
| **Value grid** | A value grid is a heatmap where each cell color shows the mean planner value assigned to that map position under fixed key and door conditions. The x-axis is grid column index, and the y-axis is grid row index. |
| **Shaped reward** | Shaped reward is the training signal used by Q-learning after adding custom bonuses and penalties to the raw environment reward. |

### Current tracked evaluation summary

The table below describes the **currently checked-in artifacts** in this repository.

| Algorithm | Environment | Evaluation episodes | Success rate | Mean raw reward | Mean steps |
| --- | --- | --- | --- | --- | --- |
| Value Iteration | `MiniGrid-DoorKey-8x8-v0` | `100` | `100.00%` | `0.976` | `16.76` |
| Q-Learning | `MiniGrid-Unlock-v0` | `100` | `99.00%` | `0.959` | `12.73` |
| Q-Learning | `MiniGrid-DoorKey-8x8-v0` | `100` | `95.00%` | `0.907` | `62.68` |
| Q-Learning | `MiniGrid-Dynamic-Obstacles-8x8-v0` | `100` | `98.00%` | `0.899` | `17.60` |

<a id="environments"></a>

## 🎮 Environments Used in GreedyMiniGrid

MiniGrid is a family of small grid-world control tasks where the agent must move, interact with objects, and reach goals. The official project page is available at [minigrid.farama.org](https://minigrid.farama.org/).

### Environment catalog

| Environment | Used by | What the agent must do |
| --- | --- | --- |
| `MiniGrid-Unlock-v0` | Q-Learning | Pick up the key, unlock the door, and reach the goal. |
| `MiniGrid-DoorKey-8x8-v0` | Value Iteration and Q-Learning | Solve the key-door-goal task on an `8×8` map. |
| `MiniGrid-Dynamic-Obstacles-8x8-v0` | Q-Learning | Reach the goal while avoiding moving obstacles in an `8×8` map. |

### Why these games are useful together

- `Unlock` and `DoorKey` test multi-step interaction with keys and doors.
- `Dynamic Obstacles` tests reactive behavior in a changing scene.
- `DoorKey` is ideal for comparing a model-based planner and a model-free learner on the same task family.

<a id="dependencies"></a>

## 🧰 Main Dependencies and Project Entry Points

### Main runtime dependencies

| Dependency | Definition |
| --- | --- |
| `uv` | The environment and package manager used to install dependencies and run commands. |
| `gymnasium` | The environment API used for resets, steps, rewards, termination flags, and rendering. |
| `minigrid` | The environment package that provides the MiniGrid tasks and wrappers. |
| `numpy` | The array library used for state encoding, reward shaping logic, and value updates. |
| `matplotlib` | The plotting library used to generate training and evaluation figures. |
| `Pillow` | The image library used to save animated GIFs. |
| `tqdm` | The progress-bar library used during long training and evaluation loops. |

### Key files

| File | Purpose |
| --- | --- |
| [RUNME.md](RUNME.md) | Installation steps, exact commands, runtime notes, and troubleshooting. |
| [PROMPTS.md](PROMPTS.md) | Curated prompt log showing how the repository was iterated through planning, implementation, plotting feedback, and documentation requests. |
| [doc/report.md](doc/report.md) | Generated artifact index built from the current contents of `doc/` and `artifacts/`. |

### Minimal command path

```bash
cd rl-minigrid-lab
uv sync --all-groups
uv run rl-minigrid-lab full-suite
```

For the full command list and debug variants, use [RUNME.md](RUNME.md).

<a id="mathematics"></a>

## 📐 Mathematics and Notation

This section defines the mathematical symbols before the algorithms use them.

### Symbols

- `𝕊` is the **state space**, and `s ∈ 𝕊` is one state.
- `𝔸` is the **action space**, and `a ∈ 𝔸` is one action.
- `S_t` is the random state observed at time step `t`.
- `A_t` is the random action selected at time step `t`.
- `R_t` is the random reward observed at time step `t`.
- `s'` is the **next state** reached after `(s, a)`.
- `π` is a **policy**, that is, a rule for choosing actions from states.
- `γ` is the **discount factor**, which reduces the effect of future reward.
- `α` is the **learning rate**, which scales how strongly new information updates the Q-table.
- `ε` is the **exploration probability**, that is, the probability of choosing a random action instead of the current best action.
- `V(s)` is the **value function**, meaning the expected future return from state `s`.
- `Q(s, a)` is the **action-value function**, meaning the expected future return after taking action `a` in state `s`.

### Value and action-value functions

The **value function** of a policy is the expected discounted return starting from state `s`.

```math
V_{\pi}(s) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t R_t \mid S_0 = s\right]
```

The **Q-function** is the expected discounted return when the first action is fixed to `a`.

```math
Q_{\pi}(s,a) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t R_t \mid S_0{=}s, A_0{=}a\right]
= r(s, a) + \gamma\mathbb{E}_{s' \sim p(\cdot|s,a)}\left[ V_\pi(s')\right]
```

Here, `r(s, a)` is the immediate reward received after taking action `a` in state `s`.

### Value Iteration formula

Value Iteration updates the value of each state using the **Bellman optimality principle**. In words, the value of a state equals its immediate reward plus the discounted value of the best reachable future state.

```math
V(s) = \mathbb{E}[r_s + \gamma \cdot V(s')\,|\,s]
```

If the transition model is expanded over all next states `s'`, the equation becomes:

```math
V(s) = r_s + \gamma \sum_{s'} P(s' | s) \cdot V(s')
```

After the values converge, the optimal policy is extracted by choosing the action with the best one-step lookahead value:

```math
\pi^*(s) =
\arg\max_{a}
\left[
r(s,a)
+
\gamma \mathbb{E}_{s' \sim p(\cdot|s,a)} V^*(s')
\right]
```

### ε-Greedy TD Q-Learning formula

**Temporal-Difference**, abbreviated as **TD**, means the algorithm updates a value estimate from another value estimate instead of waiting for the full future return to be observed.

In **ε-greedy** action selection:

- with probability `ε`, the agent explores by choosing a random action
- with probability `1 - ε`, the agent exploits by choosing the action with the current maximum Q-value

The Q-learning update used in this repository is:

```math
Q(s,a)\leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
```

This update says:

- start from the old estimate `Q(s, a)`
- compute a target made of immediate reward `r` plus the discounted best future estimate `γ max_{a'} Q(s', a')`
- move the old estimate toward that target using learning rate `α`

<a id="value-iteration"></a>

## 🧠 Value Iteration

Value Iteration is the **planner** in GreedyMiniGrid. It is used only on `MiniGrid-DoorKey-8x8-v0`.

### What the planner actually observes

The planner does not use a learned neural representation. Instead, it works with an explicit **planning state**:

| Symbol or field | Definition |
| --- | --- |
| `x` | Agent x-coordinate on the grid. |
| `y` | Agent y-coordinate on the grid. |
| `d` | Agent direction index. |
| `has_key` | Boolean flag showing whether the agent is carrying the key. |
| `door_open` | Boolean flag showing whether the door is open. |

So the planner state is the tuple `(x, y, d, has_key, door_open)`.

### Action definition

| Action | Definition |
| --- | --- |
| `forward` | Move one cell ahead if the next cell is valid. |
| `left` | Rotate the agent left by 90 degrees. |
| `right` | Rotate the agent right by 90 degrees. |
| `pickup` | Pick up the key if the key is directly in front of the agent. |
| `toggle` | Open the door if the door is directly in front of the agent and the agent already has the key. |

### Current tracked DoorKey planner metrics

| Metric | Value |
| --- | --- |
| Evaluation episodes | `100` |
| Success rate | `100.00%` |
| Mean raw reward | `0.976` |
| Mean steps | `16.76` |
| Mean planning iterations per evaluation seed | `16.11` |
| Value-grid seed | `0` |

### Canonical success rollout

The GIF below is the **first successful evaluation rollout**, and the three value-grid plots shown later in this section come from the **same successful seed**, namely seed `0`.

<p align="center">
  <img src="doc/value_iteration/doorkey/success_01.gif" alt="DoorKey value-iteration success rollout" width="62%" />
</p>

### Hyperparameters

| Hyperparameter | Definition | Value |
| --- | --- | --- |
| Discount factor `γ` | Weight assigned to future value | `0.95` |
| Convergence epsilon | Maximum allowed Bellman update change before stopping | `1e-4` |
| Canonical seed | Seed used for the tracked training artifact | `42` |

### Reward definition

The planner reward model is intentionally simple:

- reward = `1.0` if the `forward` action reaches the goal cell
- reward = `0.0` otherwise

This reward definition means the planner is solving a sparse-reward shortest-path style problem.

### Plot gallery

<table>
  <tr>
    <td width="50%" valign="top">
      <img src="doc/value_iteration/doorkey/convergence_delta.png" alt="Value iteration convergence delta plot" width="100%" />
      <p><strong>Convergence delta.</strong> X-axis = Value Iteration sweep index. Y-axis = maximum absolute change in any state value during that sweep. The falling curve shows that the value function stabilizes as the Bellman updates converge.</p>
    </td>
    <td width="50%" valign="top">
      <img src="doc/value_iteration/doorkey/eval_success_rate.png" alt="Value iteration evaluation success rate plot" width="100%" />
      <p><strong>Evaluation success rate.</strong> X-axis = evaluation episode index from 1 to 100. Y-axis = cumulative success rate, where success means positive raw environment reward. The curve stays at the top because the tracked planner solves all evaluation episodes in the current artifact set.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="doc/value_iteration/doorkey/eval_reward_distribution.png" alt="Value iteration evaluation reward distribution plot" width="100%" />
      <p><strong>Evaluation reward distribution.</strong> X-axis = raw environment reward per episode. Y-axis = density. The concentration near high reward values shows that successful solutions usually reach the goal in relatively few steps.</p>
    </td>
    <td width="50%" valign="top">
      <img src="doc/value_iteration/doorkey/value_grid_has_key_false_door_open_false.png" alt="Value grid without key and with closed door" width="100%" />
      <p><strong>Value grid: no key, closed door.</strong> X-axis = grid column index. Y-axis = grid row index. Cell color and cell text = mean state value at that location with `has_key=False` and `door_open=False`. This plot shows where the planner prefers to move before the key has been collected.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="doc/value_iteration/doorkey/value_grid_has_key_true_door_open_false.png" alt="Value grid with key and closed door" width="100%" />
      <p><strong>Value grid: key collected, door still closed.</strong> X-axis = grid column index. Y-axis = grid row index. Cell color and cell text = mean state value at that location with `has_key=True` and `door_open=False`. The high-value corridor shifts toward the door because unlocking becomes the dominant subgoal.</p>
    </td>
    <td width="50%" valign="top">
      <img src="doc/value_iteration/doorkey/value_grid_has_key_true_door_open_true.png" alt="Value grid with key and open door" width="100%" />
      <p><strong>Value grid: key collected, door open.</strong> X-axis = grid column index. Y-axis = grid row index. Cell color and cell text = mean state value at that location with `has_key=True` and `door_open=True`. The value surface now points directly toward the goal because the door constraint has already been removed.</p>
    </td>
  </tr>
</table>

<a id="epsilon-greedy-td-q-learning"></a>

## 📚 ε-Greedy TD Q-Learning

GreedyMiniGrid uses **tabular ε-Greedy TD Q-Learning** as the learning-based baseline.

### Why the Q-table is stored in a Python dictionary

The Q-table is stored as a Python **dictionary**, that is, a mapping from a **hashable state key** to a NumPy vector of action values.

| Term | Definition |
| --- | --- |
| **Hashable state key** | A tuple created by flattening the observation tensor and appending the direction value. Because tuples are hashable, they can be used as dictionary keys. |
| **Zero initialization** | When a state key appears for the first time, the repository creates a vector of zeros for all allowed actions in that environment. |
| **Average-case `O(1)` lookup** | Dictionary lookup is constant time on average, so the algorithm can update or query a state quickly even as the table grows. |

This design is useful because MiniGrid exploration is **sparse**: the learner only allocates memory for states it actually encounters. That is cleaner than preallocating an enormous dense table for many impossible or never-visited states.

### Shared learning idea

For all Q-learning tasks, the algorithm:

1. Encodes the current observation as a hashable state key.
2. Adds the state to the dictionary if the state is new.
3. Chooses an action using ε-greedy exploration.
4. Steps the environment, receives reward, and computes the next encoded state.
5. Updates `Q(s, a)` with the TD target.

---

### 🔓 Q-Learning on `MiniGrid-Unlock-v0`

`MiniGrid-Unlock-v0` asks the agent to pick up the key, open the door, and then reach the goal. The learner receives a **fully observable grid** through `FullyObsWrapper`, so it can see the entire map during training.

#### Action definition

| Action id | Action | Definition |
| --- | --- | --- |
| `0` | `left` | Rotate the agent left by 90 degrees. |
| `1` | `right` | Rotate the agent right by 90 degrees. |
| `2` | `forward` | Move one cell ahead if possible. |
| `3` | `pickup` | Pick up the object directly in front of the agent. |
| `5` | `toggle` | Interact with the object directly in front of the agent, especially the door. |

#### Observation definition

| Field | Definition |
| --- | --- |
| `image` | A full-grid tensor with three channels: object type, object color, and object state. |
| `direction` | The discrete heading of the agent. |
| Encoded state key | `tuple(image.flatten()) + (direction,)` |

#### Success GIFs

<p align="center">
  <img src="doc/q_learning/unlock/success_01.gif" alt="Unlock Q-learning success rollout 1" width="45%" />
  <img src="doc/q_learning/unlock/success_02.gif" alt="Unlock Q-learning success rollout 2" width="45%" />
</p>

#### Current tracked evaluation metrics

| Metric | Value |
| --- | --- |
| Evaluation episodes | `100` |
| Success rate | `99.00%` |
| Mean raw reward | `0.959` |
| Mean steps | `12.73` |

#### Hyperparameters

| Hyperparameter | Definition | Value |
| --- | --- | --- |
| Discount factor `γ` | Weight assigned to future reward | `0.99` |
| Learning rate `α` | Update size for each TD target | `0.90` |
| Exploration probability `ε` | Probability of selecting a random action during training | `0.20` |
| Inference exploration probability | Probability of random action during evaluation rollout | `0.01` |
| Training episodes | Number of episodes in the tracked configuration | `30000` |

#### Reward shaping

Training reward in this environment means **raw environment reward plus custom shaping**:

| Rule | Shaping term |
| --- | --- |
| Base step penalty | `-0.2` |
| Key count decreases, which means the key was picked up | `+1.0` |
| `forward` changes nothing | `-2.0` |
| `pickup` is used while the front cell is not a key | `-2.0` |
| `toggle` changes nothing | `-2.0` |

#### Plot gallery

<table>
  <tr>
    <td width="50%" valign="top">
      <img src="doc/q_learning/unlock/reward.png" alt="Unlock reward during training" width="100%" />
      <p><strong>Reward during training.</strong> X-axis = training episode index. Y-axis = cumulative shaped reward inside that episode. The thin line shows raw per-episode values, and the thick line shows the 100-episode rolling mean. The curve becomes more stable as the agent learns a reliable key-door-goal sequence.</p>
    </td>
    <td width="50%" valign="top">
      <img src="doc/q_learning/unlock/eval_success_rate.png" alt="Unlock evaluation success rate" width="100%" />
      <p><strong>Evaluation success rate.</strong> X-axis = evaluation episode index from 1 to 100. Y-axis = cumulative success rate. A success means positive raw environment reward, so the near-flat line close to `1.0` indicates very consistent task completion.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="doc/q_learning/unlock/eval_reward_distribution.png" alt="Unlock evaluation reward distribution" width="100%" />
      <p><strong>Evaluation reward distribution.</strong> X-axis = raw environment reward collected in an evaluation episode. Y-axis = density. The mass near high reward values shows that successful runs usually finish efficiently.</p>
    </td>
    <td width="50%" valign="top">
      <img src="doc/q_learning/unlock/q_table_size.png" alt="Unlock Q-table size growth" width="100%" />
      <p><strong>Q-table size.</strong> X-axis = training episode index. Y-axis = number of unique encoded state keys stored in the dictionary after that episode. Growth slows down once the agent repeatedly revisits the most useful part of the state space.</p>
    </td>
  </tr>
  <tr>
    <td colspan="2" valign="top">
      <img src="doc/q_learning/unlock/episode_length.png" alt="Unlock episode length during training" width="100%" />
      <p><strong>Episode length.</strong> X-axis = training episode index. Y-axis = number of environment steps before termination or truncation. The falling curve means the agent reaches the goal with fewer wasted actions as learning progresses.</p>
    </td>
  </tr>
</table>

---

### 🚪 Q-Learning on `MiniGrid-DoorKey-8x8-v0`

`MiniGrid-DoorKey-8x8-v0` is a larger key-door-goal task on an `8×8` grid. The learner again uses `FullyObsWrapper`, so the entire map is visible in the training observation.

#### Action definition

| Action id | Action | Definition |
| --- | --- | --- |
| `0` | `left` | Rotate the agent left by 90 degrees. |
| `1` | `right` | Rotate the agent right by 90 degrees. |
| `2` | `forward` | Move one cell ahead if possible. |
| `3` | `pickup` | Pick up the object directly in front of the agent. |
| `5` | `toggle` | Interact with the object directly in front of the agent, especially the door. |

#### Observation definition

| Field | Definition |
| --- | --- |
| `image` | A full-grid tensor with three channels: object type, object color, and object state. |
| `direction` | The discrete heading of the agent. |
| Encoded state key | `tuple(image.flatten()) + (direction,)` |

#### Success GIFs

<p align="center">
  <img src="doc/q_learning/doorkey/success_01.gif" alt="DoorKey Q-learning success rollout 1" width="45%" />
  <img src="doc/q_learning/doorkey/success_02.gif" alt="DoorKey Q-learning success rollout 2" width="45%" />
</p>

#### Current tracked evaluation metrics

| Metric | Value |
| --- | --- |
| Evaluation episodes | `100` |
| Success rate | `95.00%` |
| Mean raw reward | `0.907` |
| Mean steps | `62.68` |

#### Hyperparameters

| Hyperparameter | Definition | Value |
| --- | --- | --- |
| Discount factor `γ` | Weight assigned to future reward | `0.99` |
| Learning rate `α` | Update size for each TD target | `0.90` |
| Exploration probability `ε` | Probability of selecting a random action during training | `0.20` |
| Inference exploration probability | Probability of random action during evaluation rollout | `0.01` |
| Training episodes | Number of episodes in the tracked configuration | `30000` |

#### Reward shaping

Training reward in this environment uses the same shaping logic as `Unlock`:

| Rule | Shaping term |
| --- | --- |
| Base step penalty | `-0.2` |
| Key count decreases, which means the key was picked up | `+1.0` |
| `forward` changes nothing | `-2.0` |
| `pickup` is used while the front cell is not a key | `-2.0` |
| `toggle` changes nothing | `-2.0` |

#### Plot gallery

<table>
  <tr>
    <td width="50%" valign="top">
      <img src="doc/q_learning/doorkey/reward.png" alt="DoorKey reward during training" width="100%" />
      <p><strong>Reward during training.</strong> X-axis = training episode index. Y-axis = cumulative shaped reward inside that episode. The thin line shows per-episode reward, and the thick line shows the 100-episode rolling mean. The slower stabilization compared with `Unlock` reflects the larger search space of the `8×8` task.</p>
    </td>
    <td width="50%" valign="top">
      <img src="doc/q_learning/doorkey/eval_success_rate.png" alt="DoorKey evaluation success rate" width="100%" />
      <p><strong>Evaluation success rate.</strong> X-axis = evaluation episode index from 1 to 100. Y-axis = cumulative success rate. The line stays high but not perfectly flat, which indicates that the learned policy is strong but still occasionally makes costly mistakes.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="doc/q_learning/doorkey/eval_reward_distribution.png" alt="DoorKey evaluation reward distribution" width="100%" />
      <p><strong>Evaluation reward distribution.</strong> X-axis = raw environment reward per evaluation episode. Y-axis = density. The wider spread compared with `Unlock` reflects more variation in how efficiently the policy solves larger layouts.</p>
    </td>
    <td width="50%" valign="top">
      <img src="doc/q_learning/doorkey/q_table_size.png" alt="DoorKey Q-table size growth" width="100%" />
      <p><strong>Q-table size.</strong> X-axis = training episode index. Y-axis = number of unique encoded state keys stored in the dictionary after that episode. The sustained growth shows that the larger grid and interaction structure expose the learner to more distinct states.</p>
    </td>
  </tr>
  <tr>
    <td colspan="2" valign="top">
      <img src="doc/q_learning/doorkey/episode_length.png" alt="DoorKey episode length during training" width="100%" />
      <p><strong>Episode length.</strong> X-axis = training episode index. Y-axis = number of environment steps before termination or truncation. The trajectory drops more slowly because the agent must learn a longer chain of correct actions before reaching the goal consistently.</p>
    </td>
  </tr>
</table>

---

### ⚽ Q-Learning on `MiniGrid-Dynamic-Obstacles-8x8-v0`

`MiniGrid-Dynamic-Obstacles-8x8-v0` uses a **local egocentric observation** instead of a full-grid observation. The learner only sees a `3×3` window because `ViewSizeWrapper` is configured with `agent_view_size = 3`.

#### Action definition

| Action id | Action | Definition |
| --- | --- | --- |
| `0` | `left` | Rotate the agent left by 90 degrees. |
| `1` | `right` | Rotate the agent right by 90 degrees. |
| `2` | `forward` | Move one cell ahead if possible. |

#### Observation definition

| Field | Definition |
| --- | --- |
| `image` | A local `3×3` tensor centered on the agent view, with channels for object type, color, and state. |
| `direction` | The discrete heading of the agent. |
| Encoded state key | `tuple(image.flatten()) + (direction,)` |

#### Success GIFs

<p align="center">
  <img src="doc/q_learning/dynamic_obstacles/success_01.gif" alt="Dynamic obstacles Q-learning success rollout 1" width="45%" />
  <img src="doc/q_learning/dynamic_obstacles/success_02.gif" alt="Dynamic obstacles Q-learning success rollout 2" width="45%" />
</p>

#### Current tracked evaluation metrics

| Metric | Value |
| --- | --- |
| Evaluation episodes | `100` |
| Success rate | `98.00%` |
| Mean raw reward | `0.899` |
| Mean steps | `17.60` |

#### Hyperparameters

| Hyperparameter | Definition | Value |
| --- | --- | --- |
| Discount factor `γ` | Weight assigned to future reward | `0.98` |
| Learning rate `α` | Update size for each TD target | `0.95` |
| Exploration probability `ε` | Probability of selecting a random action during training | `0.50` |
| Inference exploration probability | Probability of random action during evaluation rollout | `0.01` |
| Training episodes | Number of episodes in the tracked configuration | `30000` |
| View size | Size of the local egocentric image window | `3` |

#### Reward shaping

Training reward in this environment multiplies the raw environment reward by `10` and then adds local movement bonuses and penalties:

| Rule | Shaping term |
| --- | --- |
| Base step penalty | `-0.1` |
| Raw environment reward multiplier | `10 × raw reward` |
| `forward` into empty traversable space with visible movement | `+0.5` |
| `forward` into the goal with visible movement | `+10.0` |
| `forward` into a wall or obstacle without movement | `-2.0` |
| `left` or `right` while forward space is already traversable | `-1.0` |

#### Plot gallery

<table>
  <tr>
    <td width="50%" valign="top">
      <img src="doc/q_learning/dynamic_obstacles/reward.png" alt="Dynamic obstacles reward during training" width="100%" />
      <p><strong>Reward during training.</strong> X-axis = training episode index. Y-axis = cumulative shaped reward inside that episode. The thin line shows per-episode reward, and the thick line shows the 100-episode rolling mean. The sharper rise reflects strong local movement incentives that help the agent react quickly.</p>
    </td>
    <td width="50%" valign="top">
      <img src="doc/q_learning/dynamic_obstacles/eval_success_rate.png" alt="Dynamic obstacles evaluation success rate" width="100%" />
      <p><strong>Evaluation success rate.</strong> X-axis = evaluation episode index from 1 to 100. Y-axis = cumulative success rate. The line remains close to the top, which indicates that the learned policy generalizes well to the evaluation seeds used in the current artifact set.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="doc/q_learning/dynamic_obstacles/eval_reward_distribution.png" alt="Dynamic obstacles evaluation reward distribution" width="100%" />
      <p><strong>Evaluation reward distribution.</strong> X-axis = raw environment reward per evaluation episode. Y-axis = density. The concentrated peak indicates that most successful rollouts finish with similar efficiency even though obstacles move dynamically.</p>
    </td>
    <td width="50%" valign="top">
      <img src="doc/q_learning/dynamic_obstacles/q_table_size.png" alt="Dynamic obstacles Q-table size growth" width="100%" />
      <p><strong>Q-table size.</strong> X-axis = training episode index. Y-axis = number of unique encoded state keys stored in the dictionary after that episode. Even with a local observation, the table still grows because moving obstacles generate many distinct view configurations.</p>
    </td>
  </tr>
  <tr>
    <td colspan="2" valign="top">
      <img src="doc/q_learning/dynamic_obstacles/episode_length.png" alt="Dynamic obstacles episode length during training" width="100%" />
      <p><strong>Episode length.</strong> X-axis = training episode index. Y-axis = number of environment steps before termination or truncation. The gradual reduction shows that the policy learns to avoid dithering and to approach the goal more directly.</p>
    </td>
  </tr>
</table>

<a id="artifact-gallery"></a>

## 🖼️ Artifact Gallery

This section is a compact visual index. The full generated artifact listing is available in [doc/report.md](doc/report.md).

<table>
  <tr>
    <td width="50%" valign="top">
      <img src="doc/q_learning/unlock/reward.png" alt="Unlock training reward preview" width="100%" />
      <p><strong>Preview: Unlock reward during training.</strong> X-axis = training episode index. Y-axis = cumulative shaped reward inside the episode. This thumbnail points to the full explanation in the `Unlock` subsection above.</p>
    </td>
    <td width="50%" valign="top">
      <img src="doc/value_iteration/doorkey/value_grid_has_key_true_door_open_true.png" alt="DoorKey value grid preview" width="100%" />
      <p><strong>Preview: DoorKey value grid.</strong> X-axis = grid column index. Y-axis = grid row index. Cell color and cell text = mean planner value for that state slice. The detailed interpretation appears in the Value Iteration section above.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="doc/q_learning/dynamic_obstacles/success_01.gif" alt="Dynamic obstacles success preview" width="100%" />
      <p><strong>Preview: Dynamic Obstacles success rollout.</strong> This animation shows one successful evaluation episode of the learned policy in a moving-obstacle environment.</p>
    </td>
    <td width="50%" valign="top">
      <img src="doc/value_iteration/doorkey/success_01.gif" alt="DoorKey planner success preview" width="100%" />
      <p><strong>Preview: DoorKey planner rollout.</strong> This animation shows the first successful planner evaluation episode, which is also the rollout used to align the value-grid figures.</p>
    </td>
  </tr>
</table>

<a id="commands"></a>

## ⚙️ Commands

The operational guide is intentionally kept in [RUNME.md](RUNME.md). The three most important commands are:

```bash
uv sync --all-groups
uv run rl-minigrid-lab full-suite
uv run rl-minigrid-lab render-report
```

Use [RUNME.md](RUNME.md) for:

- per-environment commands
- debug runs with reduced episode counts
- output paths
- troubleshooting for rendering and GIF generation

<a id="prompt-log"></a>

## 💬 Prompt Log

This repository was built iteratively from planning prompts, implementation prompts, visual feedback prompts, and presentation prompts. The curated prompt history is stored in [PROMPTS.md](PROMPTS.md).

### What is inside the prompt log

- the original project-build request, translated into a concise English summary
- the implementation instruction that turned the plan into code
- the plotting feedback that refined the evaluation figures and value grids
- the README presentation request that produced this large documentation page

<a id="references"></a>

## 📚 References

- [MiniGrid official website](https://minigrid.farama.org/)
- [MiniGrid basic usage documentation](https://minigrid.farama.org/content/basic_usage/)
- Sutton & Barto, *Reinforcement Learning: An Introduction*
- Bellman-style dynamic programming and tabular Q-learning formulations summarized from the course materials already present in this repository
